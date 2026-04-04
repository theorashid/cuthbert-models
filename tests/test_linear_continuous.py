import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg
import pytest
from _helpers import simulate_lgssm

from cuthbert_models import (
    Discretizer,
    Filter,
    Kalman,
    LinearContinuousSSM,
    LinearGaussianSSM,
    Particle,
    VanLoan,
    infer,
    smooth,
)


def _ou_params(dt: float = 0.1):
    """Scalar Ornstein-Uhlenbeck: dz = -lambda z dt + sigma dW."""
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

    # Exact discretisation for reference
    F = jax.scipy.linalg.expm(A * dt)
    Q = (sigma**2 / (2 * lam)) * (1.0 - jnp.exp(-2 * lam * dt)) * jnp.eye(state_dim)

    return m0, P0, A, L, H, R, F, Q


def _build_continuous(m0, P0, A, L, H, R):
    return LinearContinuousSSM(
        initial_mean=m0,
        initial_covariance=P0,
        drift_matrix=lambda _t: A,
        diffusion_coefficient=lambda _t: L,
        emission_weights=lambda _t: H,
        emission_covariance=lambda _t: R,
    )


def _build_discrete(m0, P0, F, Q, H, R):
    return LinearGaussianSSM(
        initial_mean=m0,
        initial_covariance=P0,
        dynamics_weights=lambda _t: F,
        dynamics_covariance=lambda _t: Q,
        emission_weights=lambda _t: H,
        emission_covariance=lambda _t: R,
    )


def _simulate_ou(key, dt=0.1, n_time=50):
    m0, P0, A, L, H, R, F, Q = _ou_params(dt)
    _, emissions = simulate_lgssm(m0, F, Q, H, R, n_time, key)
    obs_times = jnp.arange(n_time, dtype=jnp.float32) * dt
    return m0, P0, A, L, H, R, F, Q, obs_times, emissions


KALMAN_METHODS = [Kalman(), Kalman(parallel=True)]
KALMAN_IDS = ["kalman", "kalman_parallel"]


@pytest.mark.parametrize("method", KALMAN_METHODS, ids=KALMAN_IDS)
def test_log_likelihood_is_finite(method):
    m0, P0, A, L, H, R, _, _, obs_times, emissions = _simulate_ou(jr.key(0))
    model = _build_continuous(m0, P0, A, L, H, R)
    with Filter(method), Discretizer(VanLoan()):
        ll = infer(model, emissions, obs_times=obs_times).marginal_log_likelihood
    assert jnp.isfinite(ll)


@pytest.mark.parametrize("method", KALMAN_METHODS, ids=KALMAN_IDS)
def test_matches_hand_discretised(method):
    """Continuous model should match hand-discretised LGSSM."""
    dt = 0.1
    m0, P0, A, L, H, R, F, Q, obs_times, emissions = _simulate_ou(jr.key(1), dt=dt)

    cont_model = _build_continuous(m0, P0, A, L, H, R)
    disc_model = _build_discrete(m0, P0, F, Q, H, R)

    with Filter(method):
        with Discretizer(VanLoan()):
            cont_result = infer(cont_model, emissions, obs_times=obs_times)
        disc_result = infer(disc_model, emissions)

    assert jnp.allclose(
        cont_result.marginal_log_likelihood,
        disc_result.marginal_log_likelihood,
        atol=1e-3,
    )
    assert jnp.allclose(
        cont_result.filtered_means, disc_result.filtered_means, atol=1e-3,
    )
    assert jnp.allclose(
        cont_result.filtered_covariances, disc_result.filtered_covariances, atol=1e-3,
    )


@pytest.mark.parametrize("method", KALMAN_METHODS, ids=KALMAN_IDS)
def test_covariance_is_symmetric_and_positive(method):
    m0, P0, A, L, H, R, _, _, obs_times, emissions = _simulate_ou(jr.key(2))
    model = _build_continuous(m0, P0, A, L, H, R)
    with Filter(method), Discretizer(VanLoan()):
        covs = infer(model, emissions, obs_times=obs_times).filtered_covariances
    batch_transpose = jnp.transpose(covs, (0, 2, 1))
    assert jnp.allclose(covs, batch_transpose, atol=1e-8)
    eigenvals = jnp.linalg.eigvalsh(covs)
    assert jnp.all(eigenvals > 0)


def test_parallel_matches_sequential():
    m0, P0, A, L, H, R, _, _, obs_times, emissions = _simulate_ou(jr.key(3))
    model = _build_continuous(m0, P0, A, L, H, R)
    with Filter(Kalman()), Discretizer(VanLoan()):
        seq = infer(model, emissions, obs_times=obs_times)
    with Filter(Kalman(parallel=True)), Discretizer(VanLoan()):
        par = infer(model, emissions, obs_times=obs_times)
    assert jnp.allclose(seq.filtered_means, par.filtered_means, atol=1e-3)
    assert jnp.allclose(
        seq.marginal_log_likelihood, par.marginal_log_likelihood, atol=1e-2,
    )


def test_smoother_returns_smoothed_means():
    m0, P0, A, L, H, R, _, _, obs_times, emissions = _simulate_ou(jr.key(4))
    model = _build_continuous(m0, P0, A, L, H, R)
    with Filter(Kalman()), Discretizer(VanLoan()):
        result = smooth(model, emissions, obs_times=obs_times)
    assert result.smoothed_means.shape == result.filtered_means.shape
    assert result.smoothed_covariances.shape == result.filtered_covariances.shape
    assert jnp.isfinite(result.marginal_log_likelihood)


def test_smoother_covariance_leq_filter():
    m0, P0, A, L, H, R, _, _, obs_times, emissions = _simulate_ou(jr.key(5))
    model = _build_continuous(m0, P0, A, L, H, R)
    with Filter(Kalman()), Discretizer(VanLoan()):
        result = smooth(model, emissions, obs_times=obs_times)
    diff = result.filtered_covariances - result.smoothed_covariances
    eigenvals = jnp.linalg.eigvalsh(diff)
    assert jnp.all(eigenvals > -1e-6)


def test_smoother_matches_discrete():
    dt = 0.1
    m0, P0, A, L, H, R, F, Q, obs_times, emissions = _simulate_ou(jr.key(6), dt=dt)
    with Filter(Kalman()):
        with Discretizer(VanLoan()):
            cont_result = smooth(
                _build_continuous(m0, P0, A, L, H, R),
                emissions, obs_times=obs_times,
            )
        disc_result = smooth(
            _build_discrete(m0, P0, F, Q, H, R), emissions,
        )
    assert jnp.allclose(
        cont_result.smoothed_means, disc_result.smoothed_means, atol=1e-3,
    )


def test_multivariate_ou():
    """2D coupled OU process."""
    state_dim = 2
    obs_dim = 1
    dt = 0.1
    n_time = 50

    A = jnp.array([[-0.5, 0.2], [0.1, -0.3]])
    L = jnp.array([[0.3, 0.0], [0.0, 0.2]])
    H = jnp.array([[1.0, 0.0]])
    R = jnp.eye(obs_dim) * 0.1
    m0 = jnp.zeros(state_dim)
    P0 = jnp.eye(state_dim)

    # Exact discretisation for simulation
    F = jax.scipy.linalg.expm(A * dt)
    # Van Loan for reference Q
    block = jnp.block([
        [jnp.negative(A), L @ L.T],
        [jnp.zeros((state_dim, state_dim)), A.T],
    ]) * dt
    block_exp = jax.scipy.linalg.expm(block)
    F_T = block_exp[state_dim:, state_dim:]
    Q = F_T @ block_exp[:state_dim, state_dim:]

    _, emissions = simulate_lgssm(m0, F, Q, H, R, n_time, jr.key(7))
    obs_times = jnp.arange(n_time, dtype=jnp.float32) * dt

    cont_model = _build_continuous(m0, P0, A, L, H, R)
    disc_model = _build_discrete(m0, P0, F, Q, H, R)

    with Filter(Kalman()):
        with Discretizer(VanLoan()):
            cont_ll = infer(
                cont_model, emissions, obs_times=obs_times,
            ).marginal_log_likelihood
        disc_ll = infer(disc_model, emissions).marginal_log_likelihood
    assert jnp.allclose(cont_ll, disc_ll, atol=1e-3)


def test_non_uniform_obs_times():
    """Non-uniform observation spacing should still work."""
    m0, P0, A, L, H, R, _, _ = _ou_params(dt=0.1)

    obs_times = jnp.array([0.0, 0.1, 0.3, 0.35, 0.8, 1.0, 1.5])
    n_time = len(obs_times)

    # Simulate with uniform dt then just use as data
    F_sim = jax.scipy.linalg.expm(A * 0.1)
    Q_sim = jnp.eye(1) * 0.01
    _, emissions = simulate_lgssm(m0, F_sim, Q_sim, H, R, n_time, jr.key(8))

    model = _build_continuous(m0, P0, A, L, H, R)
    with Filter(Kalman()), Discretizer(VanLoan()):
        result = infer(model, emissions, obs_times=obs_times)
    assert jnp.isfinite(result.marginal_log_likelihood)
    assert result.filtered_means.shape == (n_time, 1)


def test_particle_requires_discretizer():
    from cuthbertlib.resampling.systematic import (  # noqa: PLC0415
        resampling as systematic_resampling,
    )

    m0, P0, A, L, H, R, _, _, obs_times, emissions = _simulate_ou(jr.key(9))
    model = _build_continuous(m0, P0, A, L, H, R)
    method = Particle(
        key=jr.key(0), resampling_fn=systematic_resampling, n_particles=50,
    )
    with pytest.raises(TypeError, match="requires a Discretizer"), Filter(method):
        infer(model, emissions, obs_times=obs_times)
