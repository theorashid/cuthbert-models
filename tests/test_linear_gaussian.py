import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from _helpers import random_lgssm_args
from cuthbertlib.resampling.systematic import resampling as systematic_resampling

from cuthbert_models import EKF, Kalman, LinearGaussianSSM, Particle


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


KALMAN_METHODS = [Kalman(), Kalman(parallel=True)]
KALMAN_IDS = ["kalman", "kalman_parallel"]


def test_construct_with_callables():
    params, _ = random_lgssm_args(jr.key(0))
    model = _build_model(params)
    assert callable(model.dynamics_weights)
    F = model.dynamics_weights(jnp.array(0.0))
    assert jnp.array_equal(F, params[2])


@pytest.mark.parametrize("method", KALMAN_METHODS, ids=KALMAN_IDS)
def test_log_likelihood_is_finite(method):
    params, emissions = random_lgssm_args(jr.key(1))
    model = _build_model(params)
    ll = model.infer(emissions, method=method).marginal_log_likelihood
    assert jnp.isfinite(ll)


@pytest.mark.parametrize("method", KALMAN_METHODS, ids=KALMAN_IDS)
def test_covariance_is_symmetric_and_positive(method):
    params, emissions = random_lgssm_args(jr.key(2))
    model = _build_model(params)
    covs = model.infer(emissions, method=method).filtered_covariances
    batch_transpose = jnp.transpose(covs, (0, 2, 1))
    assert jnp.allclose(covs, batch_transpose, atol=1e-8)
    eigenvals = jnp.linalg.eigvalsh(covs)
    assert jnp.all(eigenvals > 0)


def test_parallel_matches_sequential():
    params, emissions = random_lgssm_args(jr.key(3))
    model = _build_model(params)
    seq = model.infer(emissions, method=Kalman())
    par = model.infer(emissions, method=Kalman(parallel=True))
    assert jnp.allclose(seq.filtered_means, par.filtered_means, atol=1e-3)
    assert jnp.allclose(
        seq.marginal_log_likelihood, par.marginal_log_likelihood, atol=1e-2,
    )


def test_ekf_blocked():
    params, emissions = random_lgssm_args(jr.key(4))
    model = _build_model(params)
    with pytest.raises(Exception):  # noqa: B017, PT011
        model.infer(emissions, method=EKF(linearization="taylor"))
    with pytest.raises(Exception):  # noqa: B017, PT011
        model.infer(emissions, method=EKF(linearization="moments"))


def test_invalid_method():
    params, emissions = random_lgssm_args(jr.key(5))
    model = _build_model(params)
    with pytest.raises(Exception):  # noqa: B017, PT011
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
    emissions = jr.normal(jr.key(0), (batch_size, timesteps, obs_dim))
    results = jax.vmap(lambda e: model.infer(e, method=Kalman()))(emissions)
    assert results.filtered_means.shape == (batch_size, timesteps, state_dim)
    assert results.marginal_log_likelihood.shape == (batch_size,)


def test_particle_filter_finite_log_likelihood():
    params, emissions = random_lgssm_args(jr.key(6))
    model = _build_model(params)
    method = Particle(
        key=jr.key(0),
        resampling_fn=systematic_resampling,
        n_particles=200,
    )
    result = model.infer(emissions, method=method)
    assert jnp.isfinite(result.marginal_log_likelihood)
    assert result.filtered_means.shape == (emissions.shape[0], params[0].shape[0])


def test_particle_filter_agrees_with_kalman():
    params, emissions = random_lgssm_args(jr.key(7))
    model = _build_model(params)
    kalman_ll = model.infer(emissions, method=Kalman()).marginal_log_likelihood
    pf_ll = model.infer(
        emissions,
        method=Particle(
            key=jr.key(42),
            resampling_fn=systematic_resampling,
            n_particles=500,
        ),
    ).marginal_log_likelihood
    assert jnp.allclose(kalman_ll, pf_ll, atol=15.0)


def test_beartype_catches_wrong_shape():
    with pytest.raises(Exception):  # noqa: B017, PT011
        LinearGaussianSSM(
            initial_mean=jnp.ones((2, 2)),
            initial_covariance=jnp.eye(2),
            dynamics_weights=lambda _t: jnp.eye(2),
            dynamics_covariance=lambda _t: jnp.eye(2),
            emission_weights=lambda _t: jnp.eye(2),
            emission_covariance=lambda _t: jnp.eye(2),
        )


def test_smoother_returns_smoothed_means():
    params, emissions = random_lgssm_args(jr.key(8))
    model = _build_model(params)
    result = model.smooth(emissions, method=Kalman())
    assert result.smoothed_means.shape == result.filtered_means.shape
    assert result.smoothed_covariances.shape == result.filtered_covariances.shape
    assert jnp.isfinite(result.marginal_log_likelihood)


def test_smoother_covariance_leq_filter():
    params, emissions = random_lgssm_args(jr.key(9))
    model = _build_model(params)
    result = model.smooth(emissions, method=Kalman())
    diff = result.filtered_covariances - result.smoothed_covariances
    eigenvals = jnp.linalg.eigvalsh(diff)
    assert jnp.all(eigenvals > -1e-6)


def test_parallel_smoother_matches_sequential():
    params, emissions = random_lgssm_args(jr.key(10))
    model = _build_model(params)
    seq = model.smooth(emissions, method=Kalman())
    par = model.smooth(emissions, method=Kalman(parallel=True))
    assert jnp.allclose(seq.smoothed_means, par.smoothed_means, atol=1e-3)
    assert jnp.allclose(
        seq.marginal_log_likelihood, par.marginal_log_likelihood, atol=1e-2,
    )
