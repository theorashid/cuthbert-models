from collections.abc import Callable

import equinox as eqx
from jaxtyping import Array, Float

from cuthbert_models._methods import EKF, Particle
from cuthbert_models._types import GaussianPosterior, GaussianSmoothedPosterior

Method = EKF | Particle
SmoothMethod = EKF


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
            infer_ekf_moments,
            infer_particle_gaussian,
        )

        if isinstance(method, EKF):
            if method.linearization == "moments":
                return infer_ekf_moments(self, emissions)
            return infer_ekf(self, emissions)
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
            smooth_ekf_moments,
        )

        if method.linearization == "moments":
            return smooth_ekf_moments(self, emissions)
        return smooth_ekf(self, emissions)
