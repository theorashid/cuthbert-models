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

`TrainableCovariance` requires `numpyro`, which is an optional dependency:

```sh
pip install cuthbert-models[numpyro]
```

## Install

```sh
pip install cuthbert-models
```

## Dev

```sh
uv sync --dev
uv run ruff check --fix
uv run pytest
```
