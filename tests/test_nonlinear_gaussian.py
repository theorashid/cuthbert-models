import jax.numpy as jnp
import pytest

from cuthbert_models import LinearGaussianSSM, NonlinearGaussianSSM


def _build_nonlinear_model(params):
    """Build a NonlinearGaussianSSM with linear functions (for comparison)."""
    m0, P0, F, Q, H, R = params
    return NonlinearGaussianSSM(
        initial_mean=m0,
        initial_covariance=P0,
        dynamics_fn=lambda x, _t: F @ x,
        dynamics_covariance=lambda _t: Q,
        emission_fn=lambda x, _t: H @ x,
        emission_covariance=lambda _t: R,
    )


def _build_linear_model(params):
    m0, P0, F, Q, H, R = params
    return LinearGaussianSSM(
        initial_mean=m0,
        initial_covariance=P0,
        dynamics_weights=lambda _t: F,
        dynamics_covariance=lambda _t: Q,
        emission_weights=lambda _t: H,
        emission_covariance=lambda _t: R,
    )


@pytest.mark.parametrize("method", ["ekf", "ukf"])
def test_ekf_ukf_finite_log_likelihood(noisy_motion_data, method):
    params, emissions, _ = noisy_motion_data
    model = _build_nonlinear_model(params)
    ll = model.infer(emissions, method=method).marginal_log_likelihood
    assert jnp.isfinite(ll)


@pytest.mark.parametrize("method", ["ekf", "ukf"])
def test_ekf_ukf_matches_kalman_on_linear(noisy_motion_data, method):
    """EKF/UKF on a linear model should closely match exact Kalman."""
    params, emissions, _ = noisy_motion_data
    linear = _build_linear_model(params)
    nonlinear = _build_nonlinear_model(params)

    kalman_result = linear.infer(emissions, method="kalman")
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


@pytest.mark.parametrize("method", ["ekf", "ukf"])
def test_covariance_is_symmetric_and_positive(noisy_motion_data, method):
    params, emissions, _ = noisy_motion_data
    model = _build_nonlinear_model(params)
    covs = model.infer(emissions, method=method).filtered_covariances
    batch_transpose = jnp.transpose(covs, (0, 2, 1))
    assert jnp.allclose(covs, batch_transpose, atol=1e-8)
    eigenvals = jnp.linalg.eigvalsh(covs)
    assert jnp.all(eigenvals > 0)


def test_invalid_method(noisy_motion_data):
    params, emissions, _ = noisy_motion_data
    model = _build_nonlinear_model(params)
    with pytest.raises(ValueError, match="Unknown method"):
        model.infer(emissions, method="kalman")


def test_genuinely_nonlinear():
    """Test with a simple nonlinear model (quadratic dynamics)."""
    state_dim, obs_dim = 1, 1
    model = NonlinearGaussianSSM(
        initial_mean=jnp.array([1.0]),
        initial_covariance=jnp.eye(state_dim) * 0.1,
        dynamics_fn=lambda x, _t: 0.5 * x + 25.0 * x / (1.0 + x**2),
        dynamics_covariance=lambda _t: jnp.eye(state_dim) * 10.0,
        emission_fn=lambda x, _t: x**2 / 20.0,
        emission_covariance=lambda _t: jnp.eye(obs_dim) * 1.0,
    )
    emissions = jnp.ones((20, obs_dim))

    for method in ["ekf", "ukf"]:
        result = model.infer(emissions, method=method)
        assert jnp.isfinite(result.marginal_log_likelihood)
        assert result.filtered_means.shape == (20, state_dim)
