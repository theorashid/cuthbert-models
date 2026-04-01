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
    EulerMaruyama,
    Kalman,
    LinearContinuousSSM,
    LinearGaussianSSM,
    NonlinearContinuousSSM,
    NonlinearGaussianSSM,
    Particle,
    TrainableCovariance,
    TrainableWeights,
    VanLoan,
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


# --- EKF gradients flow ---


@pytest.mark.parametrize(
    "method",
    [EKF(linearization="taylor"), EKF(linearization="moments")],
    ids=["ekf_taylor", "ekf_moments"],
)
def test_ekf_gradients_are_finite(method):
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


# --- Continuous-time numpyro SVI ---

# Continuous-time OU process: dx = -lambda x dt + sigma dW
# In dynestyx this would be:
#   with Filter(filter_config=KFConfig()):
#       dsx.sample("f", model, obs_times=t, obs_values=y)
# Here it's just:
#   posterior = model.infer(obs_times, emissions, method=Kalman())
#   numpyro.factor("ll", posterior.marginal_log_likelihood)

DT_CT = 0.1
N_TIME_CT = 100
LAM_TRUE = 0.5
SIGMA_TRUE = 0.3
A_TRUE = jnp.array([[-LAM_TRUE]])
L_TRUE = jnp.array([[SIGMA_TRUE]])
H_CT = jnp.eye(1)
R_CT = jnp.eye(1) * 0.1
M0_CT = jnp.zeros(1)
P0_CT = jnp.eye(1)


def _ct_emissions(key):
    F_disc = jax.scipy.linalg.expm(A_TRUE * DT_CT)
    Q_disc = (
        SIGMA_TRUE**2 / (2 * LAM_TRUE)
        * (1 - jnp.exp(-2 * LAM_TRUE * DT_CT))
        * jnp.eye(1)
    )
    _, e = simulate_lgssm(M0_CT, F_disc, Q_disc, H_CT, R_CT, N_TIME_CT, key)
    return e


def _numpyro_model_linear_continuous(obs_times, y, m0, P0, H, R):
    lam = numpyro.sample("lam", dist.LogNormal(0.0, 0.5))
    sigma = numpyro.sample("sigma", dist.LogNormal(0.0, 0.5))

    A = jnp.array([[-lam]])
    L = jnp.array([[sigma]])

    model = LinearContinuousSSM(
        initial_mean=m0,
        initial_covariance=P0,
        drift_matrix=lambda _t: A,
        diffusion_coefficient=lambda _t: L,
        emission_weights=lambda _t: H,
        emission_covariance=lambda _t: R,
    )
    posterior = model.infer(obs_times, y, method=Kalman(), discretization=VanLoan())
    numpyro.factor("log_likelihood", posterior.marginal_log_likelihood)


def test_numpyro_svi_linear_continuous():
    emissions = _ct_emissions(jr.key(0))
    obs_times = jnp.arange(N_TIME_CT, dtype=jnp.float32) * DT_CT

    init_values = {"lam": LAM_TRUE * 1.5, "sigma": SIGMA_TRUE * 0.8}
    guide = AutoDelta(
        _numpyro_model_linear_continuous,
        init_loc_fn=init_to_value(values=init_values),
    )
    optimiser = optax.adam(5e-3)
    svi = SVI(
        _numpyro_model_linear_continuous,
        guide, optimiser, loss=Trace_ELBO(),
    )
    args = (obs_times, emissions, M0_CT, P0_CT, H_CT, R_CT)
    svi_result = svi.run(jr.key(42), 200, *args)

    assert svi_result.losses[-1] < svi_result.losses[0]
    params = guide.median(svi_result.params)
    assert jnp.isfinite(params["lam"])
    assert jnp.isfinite(params["sigma"])
    assert params["lam"] > 0


def _numpyro_model_nonlinear_continuous(obs_times, y, m0, P0, H, R):
    lam = numpyro.sample("lam", dist.LogNormal(0.0, 0.5))
    sigma = numpyro.sample("sigma", dist.LogNormal(0.0, 0.5))

    L = jnp.array([[sigma]])

    model = NonlinearContinuousSSM(
        initial_mean=m0,
        initial_covariance=P0,
        drift=lambda x, _t: -lam * x,
        diffusion_coefficient=lambda _x, _t: L,
        emission_fn=lambda x, _t: H @ x,
        emission_covariance=lambda _x, _t: R,
    )
    posterior = model.infer(
        obs_times, y, method=EKF(linearization="taylor"),
        discretization=EulerMaruyama(),
    )
    numpyro.factor("log_likelihood", posterior.marginal_log_likelihood)


def test_numpyro_svi_nonlinear_continuous():
    emissions = _ct_emissions(jr.key(0))
    obs_times = jnp.arange(N_TIME_CT, dtype=jnp.float32) * DT_CT

    init_values = {"lam": LAM_TRUE * 1.5, "sigma": SIGMA_TRUE * 0.8}
    guide = AutoDelta(
        _numpyro_model_nonlinear_continuous,
        init_loc_fn=init_to_value(values=init_values),
    )
    optimiser = optax.adam(5e-3)
    svi = SVI(
        _numpyro_model_nonlinear_continuous,
        guide, optimiser, loss=Trace_ELBO(),
    )
    args = (obs_times, emissions, M0_CT, P0_CT, H_CT, R_CT)
    svi_result = svi.run(jr.key(42), 200, *args)

    assert svi_result.losses[-1] < svi_result.losses[0]
    params = guide.median(svi_result.params)
    assert jnp.isfinite(params["lam"])
    assert jnp.isfinite(params["sigma"])
    assert params["lam"] > 0

# --- Numpyro SVI (Kalman and EKF moments) ---


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


def test_numpyro_svi_ekf_moments():
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
    args = (emissions, EKF(linearization="moments"), M0, P0, H, R)

    svi_result = svi.run(jr.key(42), 100, *args)

    assert svi_result.losses[-1] < svi_result.losses[0]
    map_params = guide.median(svi_result.params)
    assert jnp.all(jnp.isfinite(map_params["dynamics_weights"]))


# --- Forecasting with fitted SVI model ---


def test_forecast_after_svi():
    """Fit dynamics via SVI, then forecast by appending NaN emissions."""
    emissions = _emissions(jr.key(0), n_time=100)
    horizon = 10

    init_values = {
        "dynamics_weights": F_TRUE + 0.01 * jnp.eye(STATE_DIM),
        "dynamics_covariance": jnp.eye(STATE_DIM) * 0.1,
    }
    guide = AutoDelta(
        _numpyro_model_kalman,
        init_loc_fn=init_to_value(values=init_values),
    )
    svi = SVI(_numpyro_model_kalman, guide, optax.adam(5e-3), loss=Trace_ELBO())
    svi_result = svi.run(jr.key(42), 200, emissions, M0, P0, H, R)
    params = guide.median(svi_result.params)

    F_learned = params["dynamics_weights"]
    Q_learned = params["dynamics_covariance"]

    forecast_emissions = jnp.concatenate(
        [emissions, jnp.full((horizon, OBS_DIM), jnp.nan)],
    )
    model = LinearGaussianSSM(
        initial_mean=M0,
        initial_covariance=P0,
        dynamics_weights=lambda _t: F_learned,
        dynamics_covariance=lambda _t: Q_learned,
        emission_weights=lambda _t: H,
        emission_covariance=lambda _t: R,
    )

    result = model.infer(forecast_emissions, method=Kalman())

    assert result.filtered_means.shape == (100 + horizon, STATE_DIM)
    assert jnp.all(jnp.isfinite(result.filtered_means))
    forecast_covs = result.filtered_covariances[100:]
    for i in range(1, horizon):
        assert jnp.linalg.det(forecast_covs[i]) >= jnp.linalg.det(
            forecast_covs[i - 1],
        )


# --- Differentiable particle filter ---


def test_differentiable_particle_filter():
    from cuthbertlib.resampling.autodiff import stop_gradient_decorator  # noqa: PLC0415
    from cuthbertlib.resampling.systematic import (  # noqa: PLC0415
        resampling as systematic,
    )

    diff_resampling = stop_gradient_decorator(systematic)

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
        return -model.infer(
            emissions,
            method=Particle(
                key=jr.key(42),
                n_particles=200,
                resampling_fn=diff_resampling,
            ),
        ).marginal_log_likelihood

    grad_F = jax.grad(loss)(F_TRUE)
    assert jnp.all(jnp.isfinite(grad_F))
    assert grad_F.shape == F_TRUE.shape
