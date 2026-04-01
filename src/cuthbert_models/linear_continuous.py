from collections.abc import Callable

import equinox as eqx
from jaxtyping import Array, Float

from cuthbert_models._discretize import EulerMaruyama, VanLoan
from cuthbert_models._methods import EKF, Kalman, Particle
from cuthbert_models._types import GaussianPosterior, GaussianSmoothedPosterior

Method = Kalman | Particle
Discretization = VanLoan | EulerMaruyama


class LinearContinuousSSM(eqx.Module):

    """
    Linear continuous-discrete state space model.

    x_0 ~ N(initial_mean, initial_covariance)
    dx(t) = drift_matrix(t) @ x(t) dt + diffusion_coefficient(t) dW(t)
    y_k = emission_weights(t_k) @ x(t_k) + N(0, emission_covariance(t_k))

    Discretised between observation times before filtering. The
    discretization strategy is controlled by the `discretization`
    argument to `.infer()` / `.smooth()`:
      - VanLoan: exact via the Van Loan matrix exponential.
      - EulerMaruyama: first-order approximation (delegates to
        NonlinearContinuousSSM internally).

    drift_matrix:           A(t), shape (state, state)
    diffusion_coefficient:  L(t), shape (state, bm)
    emission_weights:       H(t), shape (obs, state)
    emission_covariance:    R(t), shape (obs, obs)
    """

    initial_mean: Float[Array, " state"]
    initial_covariance: Float[Array, "state state"]
    drift_matrix: Callable[[Float[Array, ""]], Float[Array, "state state"]]
    diffusion_coefficient: Callable[[Float[Array, ""]], Float[Array, "state bm"]]
    emission_weights: Callable[[Float[Array, ""]], Float[Array, "obs state"]]
    emission_covariance: Callable[[Float[Array, ""]], Float[Array, "obs obs"]]

    def infer(
        self,
        obs_times: Float[Array, " time"],
        emissions: Float[Array, "time obs"],
        method: Method,
        discretization: Discretization,
    ) -> GaussianPosterior:
        if isinstance(discretization, EulerMaruyama) or isinstance(
            method, Particle,
        ):
            from cuthbert_models.nonlinear_continuous import (  # noqa: PLC0415
                NonlinearContinuousSSM,
            )

            _diff = self.diffusion_coefficient
            nonlinear = NonlinearContinuousSSM(
                initial_mean=self.initial_mean,
                initial_covariance=self.initial_covariance,
                drift=lambda x, t: self.drift_matrix(t) @ x,
                diffusion_coefficient=lambda _x, t: _diff(t),
                emission_fn=lambda x, t: self.emission_weights(t) @ x,
                emission_covariance=lambda _x, t: self.emission_covariance(t),
            )
            nl_method: Particle | EKF = (
                method
                if isinstance(method, Particle)
                else EKF(linearization="moments")
            )
            return nonlinear.infer(
                obs_times, emissions, method=nl_method,
                discretization=EulerMaruyama(),
            )

        from cuthbert_models._inference import (  # noqa: PLC0415
            infer_linear_continuous,
        )

        return infer_linear_continuous(
            self, obs_times, emissions, parallel=method.parallel,
        )

    def smooth(
        self,
        obs_times: Float[Array, " time"],
        emissions: Float[Array, "time obs"],
        method: Kalman,
        discretization: Discretization,
    ) -> GaussianSmoothedPosterior:
        if isinstance(discretization, EulerMaruyama):
            msg = (
                "Smoothing with EulerMaruyama is not supported "
                "for LinearContinuousSSM. Use VanLoan()."
            )
            raise TypeError(msg)

        from cuthbert_models._inference import (  # noqa: PLC0415
            smooth_linear_continuous,
        )

        return smooth_linear_continuous(
            self, obs_times, emissions, parallel=method.parallel,
        )
