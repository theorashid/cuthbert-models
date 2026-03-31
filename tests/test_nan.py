"""NaN handling: interior missing observations produce finite outputs."""

import jax.numpy as jnp
import jax.random as jr
import pytest
from _helpers import random_lgssm_args
from cuthbertlib.resampling.systematic import (
    resampling as systematic_resampling,
)

from cuthbert_models import (
    EKF,
    HMM,
    Forward,
    Kalman,
    LinearGaussianSSM,
    NonlinearGaussianSSM,
    Particle,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NAN_INDICES = [5, 12, 30]


def _inject_nans(emissions, indices):
    for i in indices:
        emissions = emissions.at[i].set(jnp.nan)
    return emissions


def _build_linear(params):
    m0, P0, F, Q, H, R = params
    return LinearGaussianSSM(
        initial_mean=m0,
        initial_covariance=P0,
        dynamics_weights=lambda _t: F,
        dynamics_covariance=lambda _t: Q,
        emission_weights=lambda _t: H,
        emission_covariance=lambda _t: R,
    )


def _build_nonlinear(params):
    m0, P0, F, Q, H, R = params
    return NonlinearGaussianSSM(
        initial_mean=m0,
        initial_covariance=P0,
        dynamics_fn=lambda x, _t: F @ x,
        dynamics_covariance=lambda _t: Q,
        emission_fn=lambda x, _t: H @ x,
        emission_covariance=lambda _x, _t: R,
    )


def _random_hmm(key, n_states=3, n_time=50):
    import jax  # noqa: PLC0415

    keys = jr.split(key, 5)
    pi0 = jax.nn.softmax(jr.normal(keys[0], (n_states,)))
    A_raw = jr.normal(keys[1], (n_states, n_states))
    A = jax.nn.softmax(A_raw, axis=-1)
    means = jr.normal(keys[2], (n_states,)) * 5.0
    std = 1.0

    def emission_log_likelihood(y, _t):
        return jnp.sum(
            -0.5 * ((y[:, None] - means[None, :]) / std) ** 2
            - jnp.log(std * jnp.sqrt(2 * jnp.pi)),
            axis=0,
        )

    model = HMM(
        initial_distribution=pi0,
        transition_matrix=lambda _t: A,
        emission_log_likelihood=emission_log_likelihood,
    )

    def step(carry, rng):
        state = carry
        k1, k2 = jr.split(rng)
        state = jr.categorical(k1, jnp.log(A[state]))
        obs = means[state] + jr.normal(k2)
        return state, jnp.atleast_1d(obs)

    init_state = jr.categorical(keys[3], jnp.log(pi0))
    _, emissions = jax.lax.scan(
        step, init_state, jr.split(keys[4], n_time),
    )
    return model, emissions


# ---------------------------------------------------------------------------
# Kalman
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parallel", [False, True], ids=["seq", "par"],
)
def test_kalman_nan_finite(parallel):
    params, emissions = random_lgssm_args(jr.key(100))
    model = _build_linear(params)
    nan_emissions = _inject_nans(emissions, NAN_INDICES)
    result = model.infer(
        nan_emissions, method=Kalman(parallel=parallel),
    )
    assert jnp.all(jnp.isfinite(result.filtered_means))
    assert jnp.isfinite(result.marginal_log_likelihood)


# ---------------------------------------------------------------------------
# EKF Taylor
# ---------------------------------------------------------------------------


def test_ekf_taylor_nan_finite():
    params, emissions = random_lgssm_args(jr.key(101))
    model = _build_nonlinear(params)
    nan_emissions = _inject_nans(emissions, NAN_INDICES)
    result = model.infer(
        nan_emissions, method=EKF(linearization="taylor"),
    )
    assert jnp.all(jnp.isfinite(result.filtered_means))
    assert jnp.isfinite(result.marginal_log_likelihood)


# ---------------------------------------------------------------------------
# EKF Moments
# ---------------------------------------------------------------------------


def test_ekf_moments_nan_finite():
    params, emissions = random_lgssm_args(jr.key(102))
    model = _build_nonlinear(params)
    nan_emissions = _inject_nans(emissions, NAN_INDICES)
    result = model.infer(
        nan_emissions, method=EKF(linearization="moments"),
    )
    assert jnp.all(jnp.isfinite(result.filtered_means))
    assert jnp.isfinite(result.marginal_log_likelihood)


# ---------------------------------------------------------------------------
# Forward (HMM)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parallel", [False, True], ids=["seq", "par"],
)
def test_forward_nan_finite(parallel):
    model, emissions = _random_hmm(jr.key(103))
    nan_emissions = _inject_nans(emissions, NAN_INDICES)
    result = model.infer(
        nan_emissions, method=Forward(parallel=parallel),
    )
    assert jnp.all(jnp.isfinite(result.filtered_probs))
    assert jnp.isfinite(result.marginal_log_likelihood)


# ---------------------------------------------------------------------------
# Particle Gaussian
# ---------------------------------------------------------------------------


def test_particle_gaussian_nan_finite():
    params, emissions = random_lgssm_args(jr.key(104))
    model = _build_nonlinear(params)
    nan_emissions = _inject_nans(emissions, NAN_INDICES)
    result = model.infer(
        nan_emissions,
        method=Particle(
            key=jr.key(0),
            resampling_fn=systematic_resampling,
            n_particles=300,
        ),
    )
    assert jnp.all(jnp.isfinite(result.filtered_means))
    assert jnp.isfinite(result.marginal_log_likelihood)


# ---------------------------------------------------------------------------
# Particle HMM
# ---------------------------------------------------------------------------


def test_particle_hmm_nan_finite():
    model, emissions = _random_hmm(jr.key(105))
    nan_emissions = _inject_nans(emissions, NAN_INDICES)
    result = model.infer(
        nan_emissions,
        method=Particle(
            key=jr.key(0),
            resampling_fn=systematic_resampling,
            n_particles=300,
        ),
    )
    assert jnp.all(jnp.isfinite(result.filtered_probs))
    assert jnp.isfinite(result.marginal_log_likelihood)
