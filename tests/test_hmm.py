import jax
import jax.numpy as jnp
import pytest

from cuthbert_models import HMM

METHODS = ["forward", "forward_parallel"]


def _build_hmm():
    """Simple 2-state HMM with Gaussian emissions."""
    K = 2
    pi0 = jnp.array([0.8, 0.2])
    A = jnp.array([[0.9, 0.1], [0.3, 0.7]])
    means = jnp.array([0.0, 5.0])
    std = 1.0

    def emission_log_likelihood(y, _t):
        # y is (obs_dim,), return (K,) log-likelihoods
        return jnp.sum(
            -0.5 * ((y[:, None] - means[None, :]) / std) ** 2
            - jnp.log(std * jnp.sqrt(2 * jnp.pi)),
            axis=0,
        )

    return HMM(
        initial_distribution=pi0,
        transition_matrix=lambda _t: A,
        emission_log_likelihood=emission_log_likelihood,
    ), K, means


@pytest.fixture(scope="module")
def hmm_data():
    model, _K, means = _build_hmm()
    key = jax.random.key(42)
    n_time = 50

    # Simulate from the HMM
    def step(carry, key):
        state = carry
        k1, k2 = jax.random.split(key)
        A = model.transition_matrix(jnp.array(0.0))
        state = jax.random.categorical(k1, jnp.log(A[state]))
        obs = means[state] + jax.random.normal(k2)
        return state, jnp.atleast_1d(obs)

    init_state = jax.random.categorical(
        key, jnp.log(model.initial_distribution)
    )
    keys = jax.random.split(jax.random.key(1), n_time)
    _, emissions = jax.lax.scan(step, init_state, keys)
    return model, emissions


@pytest.mark.parametrize("method", METHODS)
def test_log_likelihood_is_finite(hmm_data, method):
    model, emissions = hmm_data
    ll = model.infer(emissions, method=method).marginal_log_likelihood
    assert jnp.isfinite(ll)


@pytest.mark.parametrize("method", METHODS)
def test_filtered_probs_are_valid(hmm_data, method):
    model, emissions = hmm_data
    posterior = model.infer(emissions, method=method)
    probs = posterior.filtered_means  # (time, K)
    assert jnp.all(probs >= 0)
    assert jnp.allclose(probs.sum(axis=-1), 1.0, atol=1e-5)


def test_parallel_matches_sequential(hmm_data):
    model, emissions = hmm_data
    seq = model.infer(emissions, method="forward")
    par = model.infer(emissions, method="forward_parallel")
    assert jnp.allclose(seq.filtered_means, par.filtered_means, atol=1e-4)
    assert jnp.allclose(
        seq.marginal_log_likelihood, par.marginal_log_likelihood, atol=1e-2
    )


def test_invalid_method(hmm_data):
    model, emissions = hmm_data
    with pytest.raises(ValueError, match="Unknown method"):
        model.infer(emissions, method="kalman")


def test_particle_filter_hmm(hmm_data):
    model, emissions = hmm_data
    result = model.infer(
        emissions, method="particle", key=jax.random.key(0), n_particles=500,
    )
    assert jnp.isfinite(result.marginal_log_likelihood)
    forward_ll = model.infer(emissions, method="forward").marginal_log_likelihood
    assert jnp.allclose(forward_ll, result.marginal_log_likelihood, atol=5.0)


def test_deterministic_hmm():
    """An HMM where state 0 always emits y=0, state 1 always emits y=1."""
    pi0 = jnp.array([1.0, 0.0])
    A = jnp.array([[0.0, 1.0], [1.0, 0.0]])  # deterministic alternation

    def emission_log_likelihood(y, _t):
        means = jnp.array([0.0, 5.0])
        std = 0.5
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
    # Alternating observations: 5, 0, 5, 0 (starting from state 0 -> transitions to 1)
    emissions = jnp.array([[5.0], [0.0], [5.0], [0.0]])
    posterior = model.infer(emissions, method="forward")

    # After observing y=5, should be in state 1 with high probability
    assert posterior.filtered_means[0, 1] > 0.99
