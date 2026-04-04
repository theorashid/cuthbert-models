from collections.abc import Callable

import equinox as eqx
from jaxtyping import Array, Float


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
    dynamics_weights: Callable[[Float[Array, ""]], Float[Array, "state state"]]
    dynamics_covariance: Callable[[Float[Array, ""]], Float[Array, "state state"]]
    emission_weights: Callable[[Float[Array, ""]], Float[Array, "obs state"]]
    emission_covariance: Callable[[Float[Array, ""]], Float[Array, "obs obs"]]
