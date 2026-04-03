from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


@dataclass(frozen=True)
class VanLoan:
    pass


# @dataclass(frozen=True)
# class AdaptiveODE:
#     """Adaptive ODE solver discretisation via diffrax.
#
#     Numerically integrates the moment ODEs (dA/dt, dQ/dt) between
#     observation times. Works for both linear and nonlinear models,
#     but introduces numerical integration error controlled by tolerances.
#
#     Requires diffrax as an optional dependency.
#     """
#
#     dt0: float = 0.01
#     max_steps: int = 1000


@dataclass(frozen=True)
class EulerMaruyama:
    pass


# ---------------------------------------------------------------------------
# Van Loan exact discretisation (1978)
#
# The continuous-time linear SDE
#   dz = A z dt + L dW
# is exactly discretised to z_{k+1} ~ N(F z_k, Q) where
#   F = expm(A dt)
#   Q = F^{-T} @ C  (read from the block matrix exponential)
#
# Van Loan trick: expm([[-A, L L^T], [0, A^T]] * dt) gives
#   [[*, F^{-T} Q], [0, F^T]]
# so F = bottom-right block transposed, Q = F^T @ top-right block.
#
# Reference: Van Loan, "Computing integrals involving the matrix
# exponential", IEEE Trans. Automatic Control, 1978.
#
# dynestyx does NOT use Van Loan. It delegates to cd-dynamax which
# numerically integrates moment ODEs (dA/dt, dQ/dt, dC/dt) via diffrax.
# Both are mathematically equivalent for LTI systems but the ODE solver
# introduces numerical integration error. We use Van Loan for exact
# (up to floating-point) discretisation with no diffrax dependency.
# ---------------------------------------------------------------------------


def van_loan_discretise(
    A: Float[Array, "s s"],
    L: Float[Array, "s b"],
    dt: Float[Array, ""],
) -> tuple[Float[Array, "s s"], Float[Array, "s s"]]:
    """Exact discretisation of dz = A z dt + L dW via the Van Loan method."""
    s = A.shape[0]
    LLT = L @ L.T
    top = jnp.concatenate([jnp.negative(A), LLT], axis=1)
    bottom = jnp.concatenate([jnp.zeros((s, s), dtype=A.dtype), A.T], axis=1)
    block = jnp.concatenate([top, bottom], axis=0) * dt
    block_exp = jax.scipy.linalg.expm(block)
    F_T = block_exp[s:, s:]
    F = F_T.T
    Q = F_T @ block_exp[:s, s:]
    return F, Q


# ---------------------------------------------------------------------------
# Euler-Maruyama discretisation
#
# Maths follows dynestyx/discretizers.py:62-67
# (_EulerMaruyamaDiscreteEvolution.__call__._step):
#   mean = x + drift(x, t) * dt
#   cov  = L @ L^T * dt
#
# We build a NonlinearGaussianSSM whose dynamics_fn and
# dynamics_covariance encode the Euler-Maruyama step, then delegate
# to the existing EKF / particle filter infrastructure.
# ---------------------------------------------------------------------------


def euler_maruyama_to_discrete(model, obs_times: Float[Array, " time"]):
    """Convert a continuous-time SSM to NonlinearGaussianSSM via Euler-Maruyama."""
    from cuthbert_models._handlers import (  # noqa: PLC0415
        _linear_continuous_to_nonlinear,
    )
    from cuthbert_models.linear_continuous import LinearContinuousSSM  # noqa: PLC0415
    from cuthbert_models.nonlinear_gaussian import (  # noqa: PLC0415
        NonlinearGaussianSSM,
    )

    if isinstance(model, LinearContinuousSSM):
        model = _linear_continuous_to_nonlinear(model)

    T = obs_times.shape[0]
    dt_arr = jnp.concatenate([
        obs_times[1:] - obs_times[:-1],
        jnp.array([obs_times[-1] - obs_times[-2]]),
    ])

    def dynamics_fn(x, t):
        t_idx = jnp.array(t, dtype=jnp.int32)
        t_now = obs_times[jnp.clip(t_idx, 0, T - 1)]
        dt = dt_arr[jnp.clip(t_idx, 0, T - 1)]
        return x + model.drift(x, t_now) * dt

    def dynamics_covariance(t):
        t_idx = jnp.array(t, dtype=jnp.int32)
        t_now = obs_times[jnp.clip(t_idx, 0, T - 1)]
        dt = dt_arr[jnp.clip(t_idx, 0, T - 1)]
        x_dummy = jnp.zeros(model.initial_mean.shape[0], dtype=t_now.dtype)
        L = model.diffusion_coefficient(x_dummy, t_now)
        return L @ L.T * dt

    return NonlinearGaussianSSM(
        initial_mean=model.initial_mean,
        initial_covariance=model.initial_covariance,
        dynamics_fn=dynamics_fn,
        dynamics_covariance=dynamics_covariance,
        emission_fn=model.emission_fn,
        emission_covariance=model.emission_covariance,
    )

