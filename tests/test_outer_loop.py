"""Outer loop tests: optax MLE and numpyro SVI, parametrised over methods."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import optax
import pytest
from _helpers import simulate_lgssm
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.infer.initialization import init_to_value

from cuthbert_models import (
    EKF,
    UKF,
    Kalman,
    LinearGaussianSSM,
    NonlinearGaussianSSM,
    TrainableCovariance,
    TrainableWeights,
)

STATE_DIM, OBS_DIM = 2, 1
DT = 1.0
F_TRUE = jnp.array([[1.0, DT], [0.0, 1.0]])
Q_TRUE = jnp.eye(STATE_DIM) * 0.05
H = jnp.array([[1.0, 0.0]])
R = jnp.eye(OBS_DIM) * 0.1
M0 = jnp.array([0.0, 1.0])
P0 = jnp.eye(STATE_DIM)


def _emissions(key, n_time=200):
    _, e = simulate_lgssm(M0, F_TRUE, Q_TRUE, H, R, n_time, key)
    return e


# --- Optax MLE (Kalman) ---


def test_optax_mle_kalman():
    emissions = _emissions(jr.key(0))

    F_init = F_TRUE + jr.normal(jr.key(1), F_TRUE.shape) * 0.05
    model = LinearGaussianSSM(
        initial_mean=M0,
        initial_covariance=P0,
        dynamics_weights=TrainableWeights(F_init),
        dynamics_covariance=lambda _t: Q_TRUE,
        emission_weights=lambda _t: H,
        emission_covariance=lambda _t: R,
    )

    filter_spec = jax.tree.map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        lambda m: m.dynamics_weights, filter_spec, replace=True,
    )
    trainable, static = eqx.partition(model, filter_spec)

    @eqx.filter_jit
    def loss_fn(trainable):
        m = eqx.combine(trainable, static)
        return -m.infer(emissions, method=Kalman()).marginal_log_likelihood

    optimiser = optax.adam(1e-2)
    opt_state = optimiser.init(eqx.filter(trainable, eqx.is_array))

    @eqx.filter_jit
    def step(trainable, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(trainable)
        updates, opt_state = optimiser.update(grads, opt_state, trainable)
        trainable = eqx.apply_updates(trainable, updates)
        return trainable, opt_state, loss

    for _ in range(200):
        trainable, opt_state, _loss = step(trainable, opt_state)

    final_model = eqx.combine(trainable, static)
    F_learned = final_model.dynamics_weights(jnp.array(0.0))
    assert jnp.allclose(F_learned, F_TRUE, atol=0.3)


# --- Optax MLE with TrainableCovariance ---


def test_optax_mle_trainable_covariance():
    emissions = _emissions(jr.key(0))

    F_init = F_TRUE + jr.normal(jr.key(1), F_TRUE.shape) * 0.05
    Q_init = jnp.eye(STATE_DIM) * 0.1
    model = LinearGaussianSSM(
        initial_mean=M0,
        initial_covariance=P0,
        dynamics_weights=TrainableWeights(F_init),
        dynamics_covariance=TrainableCovariance(Q_init),
        emission_weights=lambda _t: H,
        emission_covariance=lambda _t: R,
    )

    filter_spec = jax.tree.map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        lambda m: (m.dynamics_weights, m.dynamics_covariance),
        filter_spec,
        replace=(True, True),
    )
    trainable, static = eqx.partition(model, filter_spec)

    @eqx.filter_jit
    def loss_fn(trainable):
        m = eqx.combine(trainable, static)
        return -m.infer(emissions, method=Kalman()).marginal_log_likelihood

    optimiser = optax.adam(1e-2)
    opt_state = optimiser.init(eqx.filter(trainable, eqx.is_array))

    @eqx.filter_jit
    def step(trainable, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(trainable)
        updates, opt_state = optimiser.update(grads, opt_state, trainable)
        trainable = eqx.apply_updates(trainable, updates)
        return trainable, opt_state, loss

    initial_loss = None
    for i in range(200):
        trainable, opt_state, loss = step(trainable, opt_state)
        if i == 0:
            initial_loss = loss

    assert loss < initial_loss

    final_model = eqx.combine(trainable, static)
    Q_learned = final_model.dynamics_covariance(jnp.array(0.0))
    eigenvals = jnp.linalg.eigvalsh(Q_learned)
    assert jnp.all(eigenvals > 0)


# --- EKF/UKF gradients flow ---


@pytest.mark.parametrize("method", [EKF(), UKF()], ids=["ekf", "ukf"])
def test_ekf_ukf_gradients_are_finite(method):
    emissions = _emissions(jr.key(0), n_time=50)

    def loss(F_param):
        model = NonlinearGaussianSSM(
            initial_mean=M0,
            initial_covariance=P0,
            dynamics_fn=lambda x, _t: F_param @ x,
            dynamics_covariance=lambda _t: Q_TRUE,
            emission_fn=lambda x, _t: H @ x,
            emission_covariance=lambda _x, _t: R,
        )
        return -model.infer(emissions, method=method).marginal_log_likelihood

    grad_F = jax.grad(loss)(F_TRUE)
    assert jnp.all(jnp.isfinite(grad_F))
    assert grad_F.shape == F_TRUE.shape


# --- Numpyro SVI (Kalman and UKF) ---


def _numpyro_model_kalman(y, m0, P0, H, R):
    state_dim = m0.shape[0]
    F = numpyro.sample(
        "dynamics_weights",
        dist.Normal(jnp.eye(state_dim), 0.5).to_event(2),
    )
    Q = numpyro.sample(
        "dynamics_covariance",
        dist.InverseWishart(
            concentration=state_dim + 1,
            scale_matrix=jnp.eye(state_dim),
        ),
    )
    model = LinearGaussianSSM(
        initial_mean=m0,
        initial_covariance=P0,
        dynamics_weights=lambda _t: F,
        dynamics_covariance=lambda _t: Q,
        emission_weights=lambda _t: H,
        emission_covariance=lambda _t: R,
    )
    posterior = model.infer(y, method=Kalman())
    numpyro.factor("log_likelihood", posterior.marginal_log_likelihood)


def _numpyro_model_nonlinear(y, method, m0, P0, H, R):
    state_dim = m0.shape[0]
    F = numpyro.sample(
        "dynamics_weights",
        dist.Normal(jnp.eye(state_dim), 0.5).to_event(2),
    )
    Q = numpyro.sample(
        "dynamics_covariance",
        dist.InverseWishart(
            concentration=state_dim + 1,
            scale_matrix=jnp.eye(state_dim),
        ),
    )
    model = NonlinearGaussianSSM(
        initial_mean=m0,
        initial_covariance=P0,
        dynamics_fn=lambda x, _t: F @ x,
        dynamics_covariance=lambda _t: Q,
        emission_fn=lambda x, _t: H @ x,
        emission_covariance=lambda _x, _t: R,
    )
    posterior = model.infer(y, method=method)
    numpyro.factor("log_likelihood", posterior.marginal_log_likelihood)


def test_numpyro_svi_kalman():
    emissions = _emissions(jr.key(0), n_time=50)

    init_values = {
        "dynamics_weights": F_TRUE + 0.01 * jnp.eye(STATE_DIM),
        "dynamics_covariance": jnp.eye(STATE_DIM) * 0.1,
    }
    guide = AutoDelta(
        _numpyro_model_kalman,
        init_loc_fn=init_to_value(values=init_values),
    )
    optimiser = optax.adam(5e-3)
    svi = SVI(_numpyro_model_kalman, guide, optimiser, loss=Trace_ELBO())
    args = (emissions, M0, P0, H, R)

    svi_result = svi.run(jr.key(42), 100, *args)

    assert svi_result.losses[-1] < svi_result.losses[0]
    map_params = guide.median(svi_result.params)
    assert jnp.all(jnp.isfinite(map_params["dynamics_weights"]))


def test_numpyro_svi_ukf():
    emissions = _emissions(jr.key(0), n_time=50)

    init_values = {
        "dynamics_weights": F_TRUE + 0.01 * jnp.eye(STATE_DIM),
        "dynamics_covariance": jnp.eye(STATE_DIM) * 0.1,
    }
    guide = AutoDelta(
        _numpyro_model_nonlinear,
        init_loc_fn=init_to_value(values=init_values),
    )
    optimiser = optax.adam(5e-3)
    svi = SVI(_numpyro_model_nonlinear, guide, optimiser, loss=Trace_ELBO())
    args = (emissions, UKF(), M0, P0, H, R)

    svi_result = svi.run(jr.key(42), 100, *args)

    assert svi_result.losses[-1] < svi_result.losses[0]
    map_params = guide.median(svi_result.params)
    assert jnp.all(jnp.isfinite(map_params["dynamics_weights"]))
