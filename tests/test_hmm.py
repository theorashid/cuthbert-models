import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from cuthbertlib.resampling.systematic import resampling as systematic_resampling

from cuthbert_models import HMM, Filter, Forward, Kalman, Particle, infer, smooth


def _random_hmm_args(key, n_states: int = 3, n_time: int = 50):
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

    def step(carry, key):
        state = carry
        k1, k2 = jr.split(key)
        state = jr.categorical(k1, jnp.log(A[state]))
        obs = means[state] + jr.normal(k2)
        return state, jnp.atleast_1d(obs)

    init_state = jr.categorical(keys[3], jnp.log(pi0))
    _, emissions = jax.lax.scan(step, init_state, jr.split(keys[4], n_time))
    return model, emissions


FORWARD_METHODS = [Forward(), Forward(parallel=True)]
FORWARD_IDS = ["forward", "forward_parallel"]


@pytest.mark.parametrize("method", FORWARD_METHODS, ids=FORWARD_IDS)
def test_log_likelihood_is_finite(method):
    model, emissions = _random_hmm_args(jr.key(20))
    with Filter(method):
        ll = infer(model, emissions).marginal_log_likelihood
    assert jnp.isfinite(ll)


@pytest.mark.parametrize("method", FORWARD_METHODS, ids=FORWARD_IDS)
def test_filtered_probs_are_valid(method):
    model, emissions = _random_hmm_args(jr.key(21))
    with Filter(method):
        posterior = infer(model, emissions)
    probs = posterior.filtered_probs
    assert jnp.all(probs >= 0)
    assert jnp.allclose(probs.sum(axis=-1), 1.0, atol=1e-5)


def test_parallel_matches_sequential():
    model, emissions = _random_hmm_args(jr.key(22))
    with Filter(Forward()):
        seq = infer(model, emissions)
    with Filter(Forward(parallel=True)):
        par = infer(model, emissions)
    assert jnp.allclose(seq.filtered_probs, par.filtered_probs, atol=1e-4)
    assert jnp.allclose(
        seq.marginal_log_likelihood, par.marginal_log_likelihood, atol=1e-2,
    )


def test_invalid_method():
    model, emissions = _random_hmm_args(jr.key(23))
    with pytest.raises(Exception), Filter(Kalman()):  # noqa: B017, PT011
        infer(model, emissions)


def test_particle_filter_hmm():
    model, emissions = _random_hmm_args(jr.key(24))
    method = Particle(
        key=jr.key(0),
        resampling_fn=systematic_resampling,
        n_particles=500,
    )
    with Filter(method):
        result = infer(model, emissions)
    assert jnp.isfinite(result.marginal_log_likelihood)
    with Filter(Forward()):
        forward_ll = infer(model, emissions).marginal_log_likelihood
    assert jnp.allclose(forward_ll, result.marginal_log_likelihood, atol=5.0)


def test_deterministic_hmm():
    pi0 = jnp.array([1.0, 0.0])
    A = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    means = jnp.array([0.0, 5.0])
    std = 0.5

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
    emissions = jnp.array([[5.0], [0.0], [5.0], [0.0]])
    with Filter(Forward()):
        posterior = infer(model, emissions)
    assert posterior.filtered_probs[0, 1] > 0.99


def test_smoother_probs_are_valid():
    model, emissions = _random_hmm_args(jr.key(25))
    with Filter(Forward()):
        result = smooth(model, emissions)
    assert jnp.all(result.smoothed_probs >= 0)
    assert jnp.allclose(result.smoothed_probs.sum(axis=-1), 1.0, atol=1e-5)
    assert jnp.isfinite(result.marginal_log_likelihood)
