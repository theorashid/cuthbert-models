from collections.abc import Callable

import equinox as eqx
from jaxtyping import Array, Float


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
