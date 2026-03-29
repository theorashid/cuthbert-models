from collections.abc import Callable

import equinox as eqx
from jaxtyping import Array, Float

from cuthbert_models._methods import EKF, UKF, Particle
from cuthbert_models._types import GaussianPosterior, GaussianSmoothedPosterior

Method = EKF | UKF | Particle
SmoothMethod = EKF | UKF


class NonlinearGaussianSSM(eqx.Module):

    """
    Nonlinear Gaussian state space model.

    x_0 ~ N(initial_mean, initial_covariance)
    x_t = dynamics_fn(x_{t-1}, t) + N(0, dynamics_covariance(t))
    y_t = emission_fn(x_t, t) + N(0, emission_covariance(x_t, t))

    emission_covariance takes (x, t) so it can be state-dependent,
    subsuming dynamax's GeneralizedGaussianSSM.
    """

    initial_mean: Float[Array, " state"]
    initial_covariance: Float[Array, "state state"]
    dynamics_fn: Callable[
        [Float[Array, " state"], Float[Array, ""]], Float[Array, " state"]
    ]
    dynamics_covariance: Callable[[Float[Array, ""]], Float[Array, "state state"]]
    emission_fn: Callable[
        [Float[Array, " state"], Float[Array, ""]], Float[Array, " obs"]
    ]
    emission_covariance: Callable[
        [Float[Array, " state"], Float[Array, ""]], Float[Array, "obs obs"]
    ]

    def infer(
        self,
        emissions: Float[Array, "time obs"],
        method: Method,
    ) -> GaussianPosterior:
        from cuthbert_models._inference import (  # noqa: PLC0415
            infer_ekf,
            infer_particle_gaussian,
            infer_ukf,
        )

        if isinstance(method, EKF):
            return infer_ekf(self, emissions)
        if isinstance(method, UKF):
            return infer_ukf(self, emissions)
        return infer_particle_gaussian(
            self, emissions, key=method.key,
            n_particles=method.n_particles, resampling_fn=method.resampling_fn,
        )

    def smooth(
        self,
        emissions: Float[Array, "time obs"],
        method: SmoothMethod,
    ) -> GaussianSmoothedPosterior:
        from cuthbert_models._inference import (  # noqa: PLC0415
            smooth_ekf,
            smooth_ukf,
        )

        if isinstance(method, EKF):
            return smooth_ekf(self, emissions)
        return smooth_ukf(self, emissions)
