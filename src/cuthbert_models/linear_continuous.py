from collections.abc import Callable

import equinox as eqx
from jaxtyping import Array, Float


class LinearContinuousSSM(eqx.Module):

    """
    Linear continuous-discrete state space model.

    x_0 ~ N(initial_mean, initial_covariance)
    dx(t) = drift_matrix(t) @ x(t) dt + diffusion_coefficient(t) dW(t)
    y_k = emission_weights(t_k) @ x(t_k) + N(0, emission_covariance(t_k))

    Discretised between observation times before filtering. Requires
    a Discretizer handler (or Filter with Kalman for VanLoan).

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
