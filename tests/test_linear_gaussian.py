import jax
import jax.numpy as jnp
import pytest

from cuthbert_models import LinearGaussianSSM

METHODS = ["kalman", "kalman_parallel"]


def _build_model(params):
    m0, P0, F, Q, H, R = params
    return LinearGaussianSSM(
        initial_mean=m0,
        initial_covariance=P0,
        dynamics_weights=lambda _t: F,
        dynamics_covariance=lambda _t: Q,
        emission_weights=lambda _t: H,
        emission_covariance=lambda _t: R,
    )


def test_construct_with_callables(linear_motion_data):
    params, _, _ = linear_motion_data
    model = _build_model(params)
    assert callable(model.dynamics_weights)
    F = model.dynamics_weights(jnp.array(0.0))
    assert jnp.array_equal(F, params[2])


@pytest.mark.parametrize("method", METHODS)
def test_recovers_noiseless_trajectory(linear_motion_data, method):
    params, emissions, true_states = linear_motion_data
    model = _build_model(params)
    posterior = model.infer(emissions, method=method)
    assert jnp.allclose(posterior.filtered_means, true_states, atol=1e-4)


@pytest.mark.parametrize("method", METHODS)
def test_log_likelihood_is_finite(noisy_motion_data, method):
    params, emissions, _ = noisy_motion_data
    model = _build_model(params)
    ll = model.infer(emissions, method=method).marginal_log_likelihood
    assert jnp.isfinite(ll)


@pytest.mark.parametrize("method", METHODS)
def test_covariance_is_symmetric_and_positive(noisy_motion_data, method):
    params, emissions, _ = noisy_motion_data
    model = _build_model(params)
    covs = model.infer(emissions, method=method).filtered_covariances
    batch_transpose = jnp.transpose(covs, (0, 2, 1))
    assert jnp.allclose(covs, batch_transpose, atol=1e-8)
    eigenvals = jnp.linalg.eigvalsh(covs)
    assert jnp.all(eigenvals > 0)


def test_parallel_matches_sequential(noisy_motion_data):
    params, emissions, _ = noisy_motion_data
    model = _build_model(params)
    seq = model.infer(emissions, method="kalman")
    par = model.infer(emissions, method="kalman_parallel")
    assert jnp.allclose(seq.filtered_means, par.filtered_means, atol=1e-3)
    assert jnp.allclose(
        seq.marginal_log_likelihood, par.marginal_log_likelihood, atol=1e-2
    )


def test_invalid_method(linear_motion_data):
    params, emissions, _ = linear_motion_data
    model = _build_model(params)
    with pytest.raises(ValueError, match="Unknown method"):
        model.infer(emissions, method="bogus")


def test_vmap_over_data():
    batch_size, timesteps, state_dim, obs_dim = 5, 10, 2, 1
    model = LinearGaussianSSM(
        initial_mean=jnp.ones(state_dim),
        initial_covariance=jnp.eye(state_dim),
        dynamics_weights=lambda _t: jnp.eye(state_dim),
        dynamics_covariance=lambda _t: jnp.eye(state_dim) * 0.1,
        emission_weights=lambda _t: jnp.zeros((obs_dim, state_dim)),
        emission_covariance=lambda _t: jnp.eye(obs_dim),
    )
    emissions = jax.random.normal(jax.random.key(0), (batch_size, timesteps, obs_dim))
    results = jax.vmap(model.infer)(emissions)
    assert results.filtered_means.shape == (batch_size, timesteps, state_dim)
    assert results.marginal_log_likelihood.shape == (batch_size,)


def test_beartype_catches_wrong_shape():
    """Pass a 2D array where a 1D initial_mean is expected. beartype should catch it."""
    with pytest.raises(Exception):  # noqa: B017, PT011
        LinearGaussianSSM(
            initial_mean=jnp.ones((2, 2)),  # wrong: 2D instead of 1D
            initial_covariance=jnp.eye(2),
            dynamics_weights=lambda _t: jnp.eye(2),
            dynamics_covariance=lambda _t: jnp.eye(2),
            emission_weights=lambda _t: jnp.eye(2),
            emission_covariance=lambda _t: jnp.eye(2),
        )
