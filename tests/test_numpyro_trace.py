"""Tests for the NumpyroTrace handler."""

import jax.numpy as jnp
import jax.random as jr
import numpyro
from _helpers import simulate_lgssm

from cuthbert_models import (
    Filter,
    Kalman,
    LinearGaussianSSM,
    NumpyroTrace,
    infer,
    smooth,
)

STATE_DIM, OBS_DIM = 2, 1
F_TRUE = jnp.array([[1.0, 1.0], [0.0, 1.0]])
Q_TRUE = jnp.eye(STATE_DIM) * 0.05
H = jnp.array([[1.0, 0.0]])
R = jnp.eye(OBS_DIM) * 0.1
M0 = jnp.array([0.0, 1.0])
P0 = jnp.eye(STATE_DIM)
N_TIME = 50


def _emissions(key):
    _, e = simulate_lgssm(M0, F_TRUE, Q_TRUE, H, R, N_TIME, key)
    return e


def _make_model():
    return LinearGaussianSSM(
        initial_mean=M0,
        initial_covariance=P0,
        dynamics_weights=lambda _t: F_TRUE,
        dynamics_covariance=lambda _t: Q_TRUE,
        emission_weights=lambda _t: H,
        emission_covariance=lambda _t: R,
    )


def _get_trace(model_fn, *args):
    return numpyro.handlers.trace(
        numpyro.handlers.seed(model_fn, rng_seed=0),
    ).get_trace(*args)


def test_filter_sites():
    emissions = _emissions(jr.key(0))

    def model_fn(y):
        with Filter(Kalman()), NumpyroTrace("f"):
            infer(_make_model(), y)

    trace = _get_trace(model_fn, emissions)

    assert trace["f_log_likelihood"]["type"] == "sample"
    assert "f_marginal_log_likelihood" in trace
    assert trace["f_filtered_means"]["value"].shape == (N_TIME, STATE_DIM)
    assert trace["f_filtered_covariances"]["value"].shape == (
        N_TIME, STATE_DIM, STATE_DIM,
    )


def test_smoother_sites():
    emissions = _emissions(jr.key(0))

    def model_fn(y):
        with Filter(Kalman()), NumpyroTrace("f"):
            smooth(_make_model(), y)

    trace = _get_trace(model_fn, emissions)

    assert "f_log_likelihood" in trace
    assert "f_smoothed_means" in trace
    assert "f_smoothed_covariances" in trace
    assert trace["f_smoothed_means"]["value"].shape == (N_TIME, STATE_DIM)


def test_deterministic_false():
    emissions = _emissions(jr.key(0))

    def model_fn(y):
        with Filter(Kalman()), NumpyroTrace("f", deterministic=False):
            infer(_make_model(), y)

    trace = _get_trace(model_fn, emissions)

    assert "f_log_likelihood" in trace
    assert "f_filtered_means" not in trace
    assert "f_filtered_covariances" not in trace
    assert "f_marginal_log_likelihood" not in trace
