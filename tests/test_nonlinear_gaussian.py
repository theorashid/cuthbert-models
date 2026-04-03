import jax.numpy as jnp
import jax.random as jr
import pytest
from _helpers import random_lgssm_args
from cuthbertlib.resampling.systematic import resampling as systematic_resampling

from cuthbert_models import (
    EKF,
    Filter,
    Kalman,
    LinearGaussianSSM,
    NonlinearGaussianSSM,
    Particle,
    infer,
    smooth,
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


METHODS = [EKF(linearization="taylor"), EKF(linearization="moments")]
METHOD_IDS = ["ekf_taylor", "ekf_moments"]


@pytest.mark.parametrize("method", METHODS, ids=METHOD_IDS)
def test_finite_log_likelihood(method):
    params, emissions = random_lgssm_args(jr.key(10))
    model = _build_nonlinear(params)
    with Filter(method):
        ll = infer(model, emissions).marginal_log_likelihood
    assert jnp.isfinite(ll)


@pytest.mark.parametrize("method", METHODS, ids=METHOD_IDS)
def test_matches_kalman_on_linear(method):
    params, emissions = random_lgssm_args(jr.key(11))
    linear = _build_linear(params)
    nonlinear = _build_nonlinear(params)
    with Filter(Kalman()):
        kalman_result = infer(linear, emissions)
    with Filter(method):
        approx_result = infer(nonlinear, emissions)
    assert jnp.allclose(
        kalman_result.marginal_log_likelihood,
        approx_result.marginal_log_likelihood,
        atol=1e-2,
    )
    assert jnp.allclose(
        kalman_result.filtered_means,
        approx_result.filtered_means,
        atol=1e-2,
    )


@pytest.mark.parametrize("method", METHODS, ids=METHOD_IDS)
def test_covariance_is_symmetric_and_positive(method):
    params, emissions = random_lgssm_args(jr.key(12))
    model = _build_nonlinear(params)
    with Filter(method):
        covs = infer(model, emissions).filtered_covariances
    batch_transpose = jnp.transpose(covs, (0, 2, 1))
    assert jnp.allclose(covs, batch_transpose, atol=1e-8)
    eigenvals = jnp.linalg.eigvalsh(covs)
    assert jnp.all(eigenvals > 0)


def test_kalman_blocked():
    params, emissions = random_lgssm_args(jr.key(13))
    model = _build_nonlinear(params)
    with pytest.raises(Exception), Filter(Kalman()):  # noqa: B017, PT011
        infer(model, emissions)


def test_invalid_method():
    params, emissions = random_lgssm_args(jr.key(14))
    model = _build_nonlinear(params)
    with pytest.raises(Exception), Filter("bogus"):  # noqa: B017, PT011
        infer(model, emissions)


@pytest.mark.parametrize("method", METHODS, ids=METHOD_IDS)
def test_state_dependent_emission_covariance(method):
    state_dim, obs_dim = 1, 1
    model = NonlinearGaussianSSM(
        initial_mean=jnp.array([1.0]),
        initial_covariance=jnp.eye(state_dim) * 0.1,
        dynamics_fn=lambda x, _t: x,
        dynamics_covariance=lambda _t: jnp.eye(state_dim) * 0.1,
        emission_fn=lambda x, _t: x,
        emission_covariance=lambda x, _t: jnp.eye(obs_dim) * (0.1 + jnp.abs(x[0])),
    )
    emissions = jnp.ones((10, obs_dim))
    with Filter(method):
        result = infer(model, emissions)
    assert jnp.isfinite(result.marginal_log_likelihood)
    assert result.filtered_means.shape == (10, state_dim)


def test_particle_filter_agrees_with_kalman():
    params, emissions = random_lgssm_args(jr.key(15))
    nonlinear = _build_nonlinear(params)
    linear = _build_linear(params)
    with Filter(Kalman()):
        kalman_ll = infer(linear, emissions).marginal_log_likelihood
    pf_method = Particle(
        key=jr.key(99), resampling_fn=systematic_resampling, n_particles=500,
    )
    with Filter(pf_method):
        pf_result = infer(nonlinear, emissions)
    assert jnp.isfinite(pf_result.marginal_log_likelihood)
    assert jnp.allclose(kalman_ll, pf_result.marginal_log_likelihood, atol=15.0)


def test_genuinely_nonlinear():
    state_dim, obs_dim = 1, 1
    model = NonlinearGaussianSSM(
        initial_mean=jnp.array([1.0]),
        initial_covariance=jnp.eye(state_dim) * 0.1,
        dynamics_fn=lambda x, _t: 0.5 * x + 25.0 * x / (1.0 + x**2),
        dynamics_covariance=lambda _t: jnp.eye(state_dim) * 10.0,
        emission_fn=lambda x, _t: x**2 / 20.0,
        emission_covariance=lambda _x, _t: jnp.eye(obs_dim) * 1.0,
    )
    emissions = jnp.ones((20, obs_dim))
    for method in METHODS:
        with Filter(method):
            result = infer(model, emissions)
        assert jnp.isfinite(result.marginal_log_likelihood)
        assert result.filtered_means.shape == (20, state_dim)


@pytest.mark.parametrize("method", METHODS, ids=METHOD_IDS)
def test_smoother_matches_kalman_on_linear(method):
    params, emissions = random_lgssm_args(jr.key(16))
    linear = _build_linear(params)
    nonlinear = _build_nonlinear(params)
    with Filter(Kalman()):
        kalman_result = smooth(linear, emissions)
    with Filter(method):
        approx_result = smooth(nonlinear, emissions)
    assert jnp.allclose(
        kalman_result.smoothed_means, approx_result.smoothed_means, atol=1e-2,
    )
