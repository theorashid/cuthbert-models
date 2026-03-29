from collections.abc import Callable
from typing import Any

import equinox as eqx
from jaxtyping import Array, Float

from cuthbert_models._methods import Kalman, Particle
from cuthbert_models._types import Posterior

Method = Kalman | Particle
_DEFAULT_METHOD = Kalman()


class LinearGaussianSSM(eqx.Module):

    """
    Linear Gaussian state space model.

    x_0 ~ N(initial_mean, initial_covariance)
    x_t = dynamics_weights(t) @ x_{t-1} + N(0, dynamics_covariance(t))
    y_t = emission_weights(t) @ x_t + N(0, emission_covariance(t))

    All parameter fields are callables (t) -> Array.
    """

    initial_mean: Float[Array, " state"]
    initial_covariance: Float[Array, "state state"]
    dynamics_weights: Callable[[Any], Float[Array, "state state"]]
    dynamics_covariance: Callable[[Any], Float[Array, "state state"]]
    emission_weights: Callable[[Any], Float[Array, "obs state"]]
    emission_covariance: Callable[[Any], Float[Array, "obs obs"]]

    def infer(
        self,
        emissions: Float[Array, "time obs"],
        method: Method = _DEFAULT_METHOD,
    ) -> Posterior:
        from cuthbert_models._inference import infer_kalman  # noqa: PLC0415

        if isinstance(method, Kalman):
            return infer_kalman(self, emissions, parallel=method.parallel)

        from cuthbert_models.nonlinear_gaussian import (  # noqa: PLC0415
            NonlinearGaussianSSM,
        )

        nonlinear = NonlinearGaussianSSM(
            initial_mean=self.initial_mean,
            initial_covariance=self.initial_covariance,
            dynamics_fn=lambda x, t: self.dynamics_weights(t) @ x,
            dynamics_covariance=self.dynamics_covariance,
            emission_fn=lambda x, t: self.emission_weights(t) @ x,
            emission_covariance=lambda _x, t: self.emission_covariance(t),
        )
        return nonlinear.infer(emissions, method=method)
