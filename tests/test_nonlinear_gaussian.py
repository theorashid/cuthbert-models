import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import Array, Float

from cuthbert_models import (
    EKF,
    UKF,
    Kalman,
    LinearGaussianSSM,
    NonlinearGaussianSSM,
    Particle,
)


def _random_lgssm_args(
    key: Array, state_dim: int = 2, obs_dim: int = 1, n_time: int = 50,
) -> tuple:
    keys = jr.split(key, 7)
    m0 = jr.normal(keys[0], (state_dim,))
    P0 = jnp.eye(state_dim) * jr.uniform(keys[1], minval=0.5, maxval=1.5)
    F = jnp.eye(state_dim) + 0.1 * jr.normal(keys[2], (state_dim, state_dim))
    Q = jnp.eye(state_dim) * jr.uniform(keys[3], minval=0.01, maxval=0.1)
    H = jr.normal(keys[4], (obs_dim, state_dim))
    R = jnp.eye(obs_dim) * jr.uniform(keys[5], minval=0.05, maxval=0.2)
    states, emissions = _simulate(m0, F, Q, H, R, n_time, keys[6])
    return (m0, P0, F, Q, H, R), states, emissions


def _simulate(
    m0: Float[Array, " s"], F, Q, H, R, n_time: int, key: Array,
) -> tuple[Float[Array, "t s"], Float[Array, "t o"]]:
    state_dim, obs_dim = F.shape[0], H.shape[0]
    def step(state, key):
        k1, k2 = jr.split(key)
        next_state = F @ state + jr.multivariate_normal(k1, jnp.zeros(state_dim), Q)
        obs = H @ next_state + jr.multivariate_normal(k2, jnp.zeros(obs_dim), R)
        return next_state, (next_state, obs)
    _, (states, emissions) = jax.lax.scan(step, m0, jr.split(key, n_time))
    return states, emissions


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


METHODS = [EKF(), UKF()]
METHOD_IDS = ["ekf", "ukf"]


@pytest.mark.parametrize("method", METHODS, ids=METHOD_IDS)
def test_finite_log_likelihood(method):
    params, _, emissions = _random_lgssm_args(jr.key(10))
    model = _build_nonlinear(params)
    ll = model.infer(emissions, method=method).marginal_log_likelihood
    assert jnp.isfinite(ll)


@pytest.mark.parametrize("method", METHODS, ids=METHOD_IDS)
def test_matches_kalman_on_linear(method):
    params, _, emissions = _random_lgssm_args(jr.key(11))
    linear = _build_linear(params)
    nonlinear = _build_nonlinear(params)
    kalman_result = linear.infer(emissions, method=Kalman())
    approx_result = nonlinear.infer(emissions, method=method)
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
    params, _, emissions = _random_lgssm_args(jr.key(12))
    model = _build_nonlinear(params)
    covs = model.infer(emissions, method=method).filtered_covariances
    batch_transpose = jnp.transpose(covs, (0, 2, 1))
    assert jnp.allclose(covs, batch_transpose, atol=1e-8)
    eigenvals = jnp.linalg.eigvalsh(covs)
    assert jnp.all(eigenvals > 0)


def test_kalman_blocked():
    params, _, emissions = _random_lgssm_args(jr.key(13))
    model = _build_nonlinear(params)
    with pytest.raises(Exception):  # noqa: B017, PT011
        model.infer(emissions, method=Kalman())


def test_invalid_method():
    params, _, emissions = _random_lgssm_args(jr.key(14))
    model = _build_nonlinear(params)
    with pytest.raises(Exception):  # noqa: B017, PT011
        model.infer(emissions, method="bogus")


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
    result = model.infer(emissions, method=method)
    assert jnp.isfinite(result.marginal_log_likelihood)
    assert result.filtered_means.shape == (10, state_dim)


def test_particle_filter_agrees_with_kalman():
    params, _, emissions = _random_lgssm_args(jr.key(15))
    nonlinear = _build_nonlinear(params)
    linear = _build_linear(params)
    kalman_ll = linear.infer(emissions, method=Kalman()).marginal_log_likelihood
    pf_result = nonlinear.infer(
        emissions, method=Particle(key=jr.key(99), n_particles=500),
    )
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
    for method in [EKF(), UKF()]:
        result = model.infer(emissions, method=method)
        assert jnp.isfinite(result.marginal_log_likelihood)
        assert result.filtered_means.shape == (20, state_dim)


@pytest.mark.parametrize("method", METHODS, ids=METHOD_IDS)
def test_smoother_matches_kalman_on_linear(method):
    params, _, emissions = _random_lgssm_args(jr.key(16))
    linear = _build_linear(params)
    nonlinear = _build_nonlinear(params)
    kalman_result = linear.smooth(emissions, method=Kalman())
    approx_result = nonlinear.smooth(emissions, method=method)
    assert jnp.allclose(
        kalman_result.smoothed_means, approx_result.smoothed_means, atol=1e-2,
    )
