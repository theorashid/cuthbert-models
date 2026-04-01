from collections.abc import Callable

import equinox as eqx
from jaxtyping import Array, Float

from cuthbert_models._discretize import EulerMaruyama, VanLoan
from cuthbert_models._methods import EKF, Particle
from cuthbert_models._types import GaussianPosterior, GaussianSmoothedPosterior

Method = EKF | Particle
SmoothMethod = EKF
Discretization = EulerMaruyama | VanLoan


class NonlinearContinuousSSM(eqx.Module):

    """
    Nonlinear continuous-discrete state space model.

    x_0 ~ N(initial_mean, initial_covariance)
    dx(t) = drift(x(t), t) dt + diffusion_coefficient(x(t), t) dW(t)
    y_k = emission_fn(x(t_k), t_k) + N(0, emission_covariance(x(t_k), t_k))

    Discretised via Euler-Maruyama between observation times, then
    filtered with EKF or particle filter.

    Maths follows dynestyx/discretizers.py:62-67
    (_EulerMaruyamaDiscreteEvolution._step):
      mean = x + drift(x, t) * dt
      cov  = L @ L^T * dt
    """

    initial_mean: Float[Array, " state"]
    initial_covariance: Float[Array, "state state"]
    drift: Callable[
        [Float[Array, " state"], Float[Array, ""]], Float[Array, " state"]
    ]
    diffusion_coefficient: Callable[
        [Float[Array, " state"], Float[Array, ""]], Float[Array, "state bm"]
    ]
    emission_fn: Callable[
        [Float[Array, " state"], Float[Array, ""]], Float[Array, " obs"]
    ]
    emission_covariance: Callable[
        [Float[Array, " state"], Float[Array, ""]], Float[Array, "obs obs"]
    ]

    def infer(
        self,
        obs_times: Float[Array, " time"],
        emissions: Float[Array, "time obs"],
        method: Method,
        discretization: Discretization,
    ) -> GaussianPosterior:
        if isinstance(discretization, VanLoan):
            msg = (
                "VanLoan discretization requires a linear model. "
                "Use EulerMaruyama() for NonlinearContinuousSSM."
            )
            raise TypeError(msg)

        from cuthbert_models._inference import (  # noqa: PLC0415
            infer_nonlinear_continuous,
        )

        return infer_nonlinear_continuous(self, obs_times, emissions, method)

    def smooth(
        self,
        obs_times: Float[Array, " time"],
        emissions: Float[Array, "time obs"],
        method: SmoothMethod,
        discretization: Discretization,
    ) -> GaussianSmoothedPosterior:
        if isinstance(discretization, VanLoan):
            msg = (
                "VanLoan discretization requires a linear model. "
                "Use EulerMaruyama() for NonlinearContinuousSSM."
            )
            raise TypeError(msg)

        from cuthbert_models._inference import (  # noqa: PLC0415
            smooth_nonlinear_continuous,
        )

        return smooth_nonlinear_continuous(self, obs_times, emissions, method)
