# cuthbert-models

The white chocolate face on [cuthbert](https://github.com/state-space-models/cuthbert)'s chocolate cake backend.

`cuthbert-models` is a model specification layer for state space models.
You define the generative model as an [Equinox](https://github.com/patrick-kidger/equinox) module, call `.infer(emissions, method=...)`, and cuthbert handles the numerics.

```python
from cuthbert_models import Kalman, LinearGaussianSSM

model = LinearGaussianSSM(
    initial_mean=m0,
    initial_covariance=P0,
    dynamics_weights=lambda t: F,
    dynamics_covariance=lambda t: Q,
    emission_weights=lambda t: H,
    emission_covariance=lambda t: R,
)
posterior = model.infer(emissions, method=Kalman())
```

The model defines the generative process.
The method chooses the inference algorithm.
cuthbert does the work.

## Design

cuthbert is a low-level library. To run a Kalman filter you call `build_filter` with callback functions (`get_init_params`, `get_dynamics_params`, `get_observation_params`), construct a `model_inputs` array, then call `cuthbert.filter`. Smoothing, EKF, UKF, and particle filters each have their own `build_filter`/`build_smoother` with different callback signatures. You manage Cholesky decompositions, time indexing, and the mapping between your model parameters and cuthbert's internal representation yourself.

cuthbert-models removes all of that. You specify the generative model once -- initial distribution, dynamics, emissions -- and the bridge layer (`_inference.py`) handles:

- Converting covariance matrices to Cholesky factors
- Building the correct callback functions for each algorithm
- Mapping time indices to your callable parameters
- Constructing `model_inputs` and calling `cuthbert.filter`/`cuthbert.smoother`
- Packaging raw filter states into typed posterior objects

The user never touches `build_filter`. Switching from Kalman to EKF to particle filtering is just changing the method argument. Because models are `eqx.Module`s, they compose with the JAX ecosystem: `jax.grad`, `jax.vmap`, `eqx.partition`, optax, numpyro all work out of the box.

## Models

Each model is an `eqx.Module` with `.infer()` and `.smooth()`.
All parameter fields are callables `(t) -> Array`, matching cuthbert's callback contract.

| Model | Structure | Infer methods | Smooth methods |
|---|---|---|---|
| `LinearGaussianSSM` | Linear dynamics, Gaussian noise | `Kalman`, `Particle` | `Kalman` |
| `NonlinearGaussianSSM` | Nonlinear dynamics/emissions, Gaussian noise | `EKF`, `UKF`, `Particle` | `EKF`, `UKF` |
| `HMM` | Discrete latent states | `Forward`, `Particle` | `Forward` |

Methods are frozen dataclasses with configuration:

```python
from cuthbert_models import EKF, UKF, Kalman, Forward, Particle

Kalman()                    # sequential
Kalman(parallel=True)       # associative scan
EKF()
UKF()
Forward()
Forward(parallel=True)
Particle(key=jax.random.key(0), n_particles=500)
```

Type checking blocks invalid combinations -- `EKF()` on `LinearGaussianSSM` or `Kalman()` on `NonlinearGaussianSSM` are rejected.

## Return types

```python
from cuthbert_models import GaussianPosterior, DiscretePosterior

# Gaussian models return GaussianPosterior / GaussianSmoothedPosterior
posterior = model.infer(emissions, method=Kalman())
posterior.marginal_log_likelihood   # Float[Array, ""]
posterior.filtered_means            # Float[Array, "time state"]
posterior.filtered_covariances      # Float[Array, "time state state"]

smoothed = model.smooth(emissions, method=Kalman())
smoothed.smoothed_means             # Float[Array, "time state"]
smoothed.smoothed_covariances       # Float[Array, "time state state"]

# HMMs return DiscretePosterior / DiscreteSmoothedPosterior
posterior = hmm.infer(emissions, method=Forward())
posterior.filtered_probs            # Float[Array, "time state"]

smoothed = hmm.smooth(emissions, method=Forward())
smoothed.smoothed_probs             # Float[Array, "time state"]
```

`GaussianSmoothedPosterior` extends `GaussianPosterior`, so `isinstance` checks work as expected.

## Inner loop, outer loop

The inner loop (filtering/smoothing) is handled by cuthbert.
The outer loop (parameter estimation) is your choice:

- **optax**: MLE via `eqx.partition`/`combine`, differentiate through `.infer()`
- **numpyro**: `numpyro.factor("ll", posterior.marginal_log_likelihood)`
- **EM**: smoother E-step + numerical M-step

The model doesn't care which outer loop you use.

### Trainable parameters

```python
from cuthbert_models import TrainableWeights, TrainableCovariance

model = LinearGaussianSSM(
    initial_mean=m0,
    initial_covariance=P0,
    dynamics_weights=TrainableWeights(F_init),
    dynamics_covariance=TrainableCovariance(Q_init),  # PSD-constrained via numpyro bijectors
    emission_weights=lambda _t: H,
    emission_covariance=lambda _t: R,
)
```

`TrainableCovariance` requires `numpyro`, which is an optional dependency (`uv sync --extra numpyro`).

## Install

Not yet on PyPI. Install from source:

```sh
git clone https://github.com/state-space-models/cuthbert-models.git
cd cuthbert-models
uv sync --dev
```

## Dev

```sh
uv run ruff check --fix
uv run pytest
```
