from collections.abc import Callable
from typing import Any

import equinox as eqx
from jaxtyping import Array, Float

from cuthbert_models._types import Posterior

VALID_METHODS = {"ekf", "ukf", "particle"}


class NonlinearGaussianSSM(eqx.Module):

    """
    Nonlinear Gaussian state space model.

    x_0 ~ N(initial_mean, initial_covariance)
    x_t = dynamics_fn(x_{t-1}, t) + N(0, dynamics_covariance(t))
    y_t = emission_fn(x_t, t) + N(0, emission_covariance(t))

    dynamics_fn and emission_fn are arbitrary nonlinear functions.
    Covariance fields are callables (t) -> Array.
    """

    initial_mean: Float[Array, " state"]
    initial_covariance: Float[Array, "state state"]
    dynamics_fn: Callable[[Any, Any], Float[Array, " state"]]
    dynamics_covariance: Callable[[Any], Float[Array, "state state"]]
    emission_fn: Callable[[Any, Any], Float[Array, " obs"]]
    emission_covariance: Callable[[Any], Float[Array, "obs obs"]]

    def infer(
        self,
        emissions: Float[Array, "time obs"],
        method: str = "ekf",
    ) -> Posterior:
        method = _resolve_method(method)

        from cuthbert_models._inference import (  # noqa: PLC0415
            infer_ekf,
            infer_ukf,
        )

        if method == "ekf":
            return infer_ekf(self, emissions)
        if method == "ukf":
            return infer_ukf(self, emissions)
        msg = "Particle filter not yet implemented for NonlinearGaussianSSM"
        raise NotImplementedError(msg)


def _resolve_method(method: str) -> str:
    if method not in VALID_METHODS:
        msg = f"Unknown method {method!r}. Valid: {sorted(VALID_METHODS)}"
        raise ValueError(msg)
    return method
