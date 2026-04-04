import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg
import pytest
from _helpers import simulate_lgssm

from cuthbert_models import (
    EKF,
    Discretizer,
    EulerMaruyama,
    Filter,
    Kalman,
    LinearContinuousSSM,
    NonlinearContinuousSSM,
    NonlinearGaussianSSM,
    VanLoan,
    infer,
    smooth,
)


def _ou_continuous(dt=0.1, n_time=50, key=None):
    """OU process as a NonlinearContinuousSSM (linear drift, for comparison)."""
    if key is None:
        key = jr.key(0)
    lam = 0.5
    sigma = 0.3
    state_dim = 1
    obs_dim = 1

    A = jnp.array([[-lam]])
    L = jnp.array([[sigma]])
    H = jnp.eye(obs_dim, state_dim)
    R = jnp.eye(obs_dim) * 0.1
    m0 = jnp.zeros(state_dim)
    P0 = jnp.eye(state_dim)

    # Simulate from exact discretisation
    F = jax.scipy.linalg.expm(A * dt)
    Q = (sigma**2 / (2 * lam)) * (1 - jnp.exp(-2 * lam * dt)) * jnp.eye(1)
    _, emissions = simulate_lgssm(m0, F, Q, H, R, n_time, key)
    obs_times = jnp.arange(n_time, dtype=jnp.float32) * dt

    model = NonlinearContinuousSSM(
        initial_mean=m0,
        initial_covariance=P0,
        drift=lambda x, _t: -lam * x,
        diffusion_coefficient=lambda _x, _t: L,
        emission_fn=lambda x, _t: H @ x,
        emission_covariance=lambda _x, _t: R,
    )
    return model, obs_times, emissions, m0, P0, A, L, H, R


EKF_METHODS = [EKF(linearization="taylor"), EKF(linearization="moments")]
EKF_IDS = ["ekf_taylor", "ekf_moments"]


@pytest.mark.parametrize("method", EKF_METHODS, ids=EKF_IDS)
def test_log_likelihood_is_finite(method):
    model, obs_times, emissions, *_ = _ou_continuous(key=jr.key(0))
    with Filter(method), Discretizer(EulerMaruyama()):
        ll = infer(model, emissions, obs_times=obs_times).marginal_log_likelihood
    assert jnp.isfinite(ll)


@pytest.mark.parametrize("method", EKF_METHODS, ids=EKF_IDS)
def test_matches_linear_continuous_on_linear_model(method):
    """EM-discretised nonlinear should approximate exact linear result."""
    dt = 0.01  # small dt so Euler-Maruyama is close to exact
    n_time = 100
    model, obs_times, emissions, m0, P0, A, L, H, R = _ou_continuous(
        dt=dt, n_time=n_time, key=jr.key(1),
    )

    linear_model = LinearContinuousSSM(
        initial_mean=m0,
        initial_covariance=P0,
        drift_matrix=lambda _t: A,
        diffusion_coefficient=lambda _t: L,
        emission_weights=lambda _t: H,
        emission_covariance=lambda _t: R,
    )

    with Filter(method), Discretizer(EulerMaruyama()):
        nl_result = infer(model, emissions, obs_times=obs_times)

    with Filter(Kalman()), Discretizer(VanLoan()):
        lin_result = infer(linear_model, emissions, obs_times=obs_times)

    assert jnp.allclose(
        nl_result.marginal_log_likelihood,
        lin_result.marginal_log_likelihood,
        atol=1.0,
    )


@pytest.mark.parametrize("method", EKF_METHODS, ids=EKF_IDS)
def test_covariance_is_symmetric_and_positive(method):
    model, obs_times, emissions, *_ = _ou_continuous(key=jr.key(2))
    with Filter(method), Discretizer(EulerMaruyama()):
        covs = infer(
            model, emissions, obs_times=obs_times,
        ).filtered_covariances
    batch_transpose = jnp.transpose(covs, (0, 2, 1))
    assert jnp.allclose(covs, batch_transpose, atol=1e-6)
    eigenvals = jnp.linalg.eigvalsh(covs)
    assert jnp.all(eigenvals > 0)


def test_matches_hand_discretised_nonlinear_gaussian():
    """EM-discretised should match manually built NonlinearGaussianSSM."""
    dt = 0.1
    n_time = 50
    lam = 0.5
    sigma = 0.3
    A = jnp.array([[-lam]])
    L = jnp.array([[sigma]])
    H = jnp.eye(1)
    R = jnp.eye(1) * 0.1
    m0 = jnp.zeros(1)
    P0 = jnp.eye(1)

    F = jax.scipy.linalg.expm(A * dt)
    Q_exact = (sigma**2 / (2 * lam)) * (1 - jnp.exp(-2 * lam * dt))
    _, emissions = simulate_lgssm(
        m0, F, jnp.eye(1) * Q_exact, H, R, n_time, jr.key(3),
    )
    obs_times = jnp.arange(n_time, dtype=jnp.float32) * dt

    cont_model = NonlinearContinuousSSM(
        initial_mean=m0,
        initial_covariance=P0,
        drift=lambda x, _t: -lam * x,
        diffusion_coefficient=lambda _x, _t: L,
        emission_fn=lambda x, _t: H @ x,
        emission_covariance=lambda _x, _t: R,
    )

    # Manually Euler-Maruyama discretised
    Q_em = L @ L.T * dt
    disc_model = NonlinearGaussianSSM(
        initial_mean=m0,
        initial_covariance=P0,
        dynamics_fn=lambda x, _t: x + (-lam * x) * dt,
        dynamics_covariance=lambda _t: Q_em,
        emission_fn=lambda x, _t: H @ x,
        emission_covariance=lambda _x, _t: R,
    )

    method = EKF(linearization="taylor")
    with Filter(method):
        with Discretizer(EulerMaruyama()):
            cont_ll = infer(
                cont_model, emissions, obs_times=obs_times,
            ).marginal_log_likelihood
        disc_ll = infer(disc_model, emissions).marginal_log_likelihood

    assert jnp.allclose(cont_ll, disc_ll, atol=1e-3)


@pytest.mark.parametrize("method", EKF_METHODS, ids=EKF_IDS)
def test_smoother_returns_smoothed_means(method):
    model, obs_times, emissions, *_ = _ou_continuous(key=jr.key(4))
    with Filter(method), Discretizer(EulerMaruyama()):
        result = smooth(model, emissions, obs_times=obs_times)
    assert result.smoothed_means.shape == result.filtered_means.shape
    assert jnp.isfinite(result.marginal_log_likelihood)


def test_genuinely_nonlinear():
    """Nonlinear drift: dz = sin(z) dt + sigma dW."""
    sigma = 0.3
    state_dim = 1
    obs_dim = 1
    dt = 0.1
    n_time = 50

    m0 = jnp.array([1.0])
    P0 = jnp.eye(state_dim)
    L = jnp.array([[sigma]])
    H = jnp.eye(obs_dim, state_dim)
    R = jnp.eye(obs_dim) * 0.1

    # Simulate with Euler-Maruyama
    def sim_step(state, key_pair):
        k1, k2 = key_pair
        drift = jnp.sin(state)
        next_state = (
            state
            + drift * dt
            + L[:, 0] * jnp.sqrt(dt) * jr.normal(k1, (state_dim,))
        )
        obs = H @ next_state + jr.normal(k2, (obs_dim,)) * jnp.sqrt(R[0, 0])
        return next_state, (next_state, obs)

    keys = jr.split(jr.key(5), (n_time, 2))
    _, (_, emissions) = jax.lax.scan(sim_step, m0, keys)
    obs_times = jnp.arange(n_time, dtype=jnp.float32) * dt

    model = NonlinearContinuousSSM(
        initial_mean=m0,
        initial_covariance=P0,
        drift=lambda x, _t: jnp.sin(x),
        diffusion_coefficient=lambda _x, _t: L,
        emission_fn=lambda x, _t: H @ x,
        emission_covariance=lambda _x, _t: R,
    )

    with Filter(EKF(linearization="taylor")), Discretizer(EulerMaruyama()):
        result = infer(model, emissions, obs_times=obs_times)
    assert jnp.isfinite(result.marginal_log_likelihood)
    assert result.filtered_means.shape == (n_time, state_dim)


def test_requires_discretizer():
    model, obs_times, emissions, *_ = _ou_continuous(key=jr.key(6))
    with (
        pytest.raises(TypeError, match="requires a Discretizer"),
        Filter(EKF(linearization="taylor")),
    ):
        infer(model, emissions, obs_times=obs_times)
