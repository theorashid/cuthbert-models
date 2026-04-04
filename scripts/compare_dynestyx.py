# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cuthbert-models",
#     "dynestyx @ git+https://github.com/BasisResearch/dynestyx.git",
#     "effectful",
#     "cd-dynamax @ git+https://github.com/HD-UQ/cd_dynamax.git@public",
#     "numpyro>=0.20.1",
#     "jax>=0.9.2",
#     "optax>=0.2.8",
# ]
#
# [tool.uv.sources]
# cuthbert-models = { path = ".." }
# cd-dynamax = { git = "https://github.com/HD-UQ/cd_dynamax.git", branch = "public" }
# ///
"""
Compare dynestyx and cuthbert-models on the same continuous-time model,
both inside numpyro model functions.

Model: scalar Ornstein-Uhlenbeck process
  dx = -lambda * x dt + sigma dW,   y_k = x(t_k) + N(0, R)

Run with:
  cd cuthbert-models
  uv run scripts/compare_dynestyx.py
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.infer.initialization import init_to_value

# -----------------------------------------------------------------------
# Ground truth parameters
# -----------------------------------------------------------------------
LAM_TRUE = 0.5
SIGMA_TRUE = 0.3
DT = 0.1
N_TIME = 100

A = jnp.array([[-LAM_TRUE]])
L = jnp.array([[SIGMA_TRUE]])
H = jnp.eye(1)
R = jnp.eye(1) * 0.1
M0 = jnp.zeros(1)
P0 = jnp.eye(1)

# Exact discretisation for simulation
F_EXACT = jax.scipy.linalg.expm(A * DT)
Q_EXACT = (
    (SIGMA_TRUE**2 / (2 * LAM_TRUE))
    * (1 - jnp.exp(-2 * LAM_TRUE * DT))
    * jnp.eye(1)
)

# -----------------------------------------------------------------------
# Simulate data
# -----------------------------------------------------------------------


def simulate(key):
    def step(state, key):
        k1, k2 = jr.split(key)
        next_state = F_EXACT @ state + jr.multivariate_normal(
            k1, jnp.zeros(1), Q_EXACT,
        )
        obs = H @ next_state + jr.multivariate_normal(k2, jnp.zeros(1), R)
        return next_state, obs

    _, emissions = jax.lax.scan(step, M0, jr.split(key, N_TIME))
    return emissions


EMISSIONS = simulate(jr.key(0))
OBS_TIMES = jnp.arange(N_TIME, dtype=jnp.float32) * DT

# -----------------------------------------------------------------------
# dynestyx: numpyro model
# -----------------------------------------------------------------------
import dynestyx as dsx  # noqa: E402, I001, RUF100
from dynestyx import (  # noqa: E402, RUF100
    ContinuousTimeStateEvolution,
    DynamicalModel,
    LinearGaussianObservation,
)
from dynestyx.models.state_evolution import AffineDrift  # noqa: E402, RUF100
from dynestyx.inference.filters import (  # noqa: E402, RUF100
    ContinuousTimeKFConfig,
    Filter as DsxFilter,
)


def dynestyx_numpyro_model(obs_times, obs_values):
    lam = numpyro.sample("lam", dist.LogNormal(0.0, 0.5))
    sigma = numpyro.sample("sigma", dist.LogNormal(0.0, 0.5))

    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(M0, P0),
        state_evolution=ContinuousTimeStateEvolution(
            drift=AffineDrift(A=jnp.array([[-lam]])),
            diffusion_coefficient=lambda _x, _u, _t: jnp.array([[sigma]]),
        ),
        observation_model=LinearGaussianObservation(H=H, R=R),
    )

    with DsxFilter(filter_config=ContinuousTimeKFConfig()):
        dsx.sample(
            "f", dynamics,
            obs_times=obs_times, obs_values=obs_values,
        )


# -----------------------------------------------------------------------
# cuthbert-models: numpyro model
# -----------------------------------------------------------------------
from cuthbert_models import (  # noqa: E402, RUF100
    Discretizer,
    Filter,
    Kalman,
    LinearContinuousSSM,
    NumpyroTrace,
    VanLoan,
    infer,
)


def cuthbert_numpyro_model(obs_times, obs_values):
    lam = numpyro.sample("lam", dist.LogNormal(0.0, 0.5))
    sigma = numpyro.sample("sigma", dist.LogNormal(0.0, 0.5))

    model = LinearContinuousSSM(
        initial_mean=M0,
        initial_covariance=P0,
        drift_matrix=lambda _t: jnp.array([[-lam]]),
        diffusion_coefficient=lambda _t: jnp.array([[sigma]]),
        emission_weights=lambda _t: H,
        emission_covariance=lambda _t: R,
    )

    with (
        Filter(Kalman()),
        Discretizer(VanLoan()),
        NumpyroTrace("ssm", deterministic=False),
    ):
        infer(model, obs_values, obs_times=obs_times)


# -----------------------------------------------------------------------
# Run both at the true parameters to compare log-likelihoods
# -----------------------------------------------------------------------

print("=== Continuous-time OU: dynestyx vs cuthbert-models ===\n") # noqa: T201

# -- dynestyx --
with numpyro.handlers.seed(rng_seed=0):  # noqa: SIM117
    with numpyro.handlers.condition(
        data={"lam": LAM_TRUE, "sigma": SIGMA_TRUE},
    ):
        with numpyro.handlers.trace() as dsx_trace:
            dynestyx_numpyro_model(OBS_TIMES, EMISSIONS)

dsx_site = dsx_trace["f_marginal_log_likelihood"]
dsx_ll = float(dsx_site["fn"].log_prob(dsx_site["value"]).sum())

# -- cuthbert-models --
with numpyro.handlers.seed(rng_seed=0):  # noqa: SIM117
    with numpyro.handlers.condition(
        data={"lam": LAM_TRUE, "sigma": SIGMA_TRUE},
    ):
        with numpyro.handlers.trace() as cm_trace:
            cuthbert_numpyro_model(OBS_TIMES, EMISSIONS)

cm_site = cm_trace["ssm_log_likelihood"]
cm_ll = float(cm_site["fn"].log_prob(cm_site["value"]).sum())

print(f"dynestyx ll:        {dsx_ll:.6f}") # noqa: T201
print(f"cuthbert-models ll: {cm_ll:.6f}") # noqa: T201
print(f"difference:         {abs(dsx_ll - cm_ll):.2e}") # noqa: T201
print() # noqa: T201

# -----------------------------------------------------------------------
# Run SVI on both and compare learned parameters
# -----------------------------------------------------------------------

import optax  # noqa: E402, RUF100

INIT_VALUES = {"lam": LAM_TRUE * 1.5, "sigma": SIGMA_TRUE * 0.8}
N_STEPS = 300

# -- dynestyx SVI --
dsx_guide = AutoDelta(
    dynestyx_numpyro_model,
    init_loc_fn=init_to_value(values=INIT_VALUES),
)
dsx_svi = SVI(
    dynestyx_numpyro_model, dsx_guide,
    optax.adam(5e-3), loss=Trace_ELBO(),
)
dsx_result = dsx_svi.run(jr.key(42), N_STEPS, OBS_TIMES, EMISSIONS)
dsx_params = dsx_guide.median(dsx_result.params)

# -- cuthbert-models SVI --
cm_guide = AutoDelta(
    cuthbert_numpyro_model,
    init_loc_fn=init_to_value(values=INIT_VALUES),
)
cm_svi = SVI(
    cuthbert_numpyro_model, cm_guide,
    optax.adam(5e-3), loss=Trace_ELBO(),
)
cm_result = cm_svi.run(jr.key(42), N_STEPS, OBS_TIMES, EMISSIONS)
cm_params = cm_guide.median(cm_result.params)

print("SVI parameter estimates:")  # noqa: T201
print(f"  {'':20s} {'dynestyx':>12s} {'cuthbert-models':>16s} {'true':>8s}") # noqa: T201
print( # noqa: T201
    f"  {'lam':20s} {float(dsx_params['lam']):12.4f}"
    f" {float(cm_params['lam']):16.4f} {LAM_TRUE:8.4f}",
)
print( # noqa: T201
    f"  {'sigma':20s} {float(dsx_params['sigma']):12.4f}"
    f" {float(cm_params['sigma']):16.4f} {SIGMA_TRUE:8.4f}",
)
print() # noqa: T201
print(f"SVI final loss  dynestyx: {dsx_result.losses[-1]:.4f}") # noqa: T201
print(f"SVI final loss  cuthbert: {cm_result.losses[-1]:.4f}") # noqa: T201
print( # noqa: T201
    f"Loss difference:          "
    f"{abs(float(dsx_result.losses[-1]) - float(cm_result.losses[-1])):.2e}",
)
