from collections.abc import Callable

import equinox as eqx
from jaxtyping import Array, Float


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
