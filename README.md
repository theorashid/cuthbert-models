# cuthbert-models

`cuthbert-models` is a model specification layer for state space models.
Define the generative model as an [Equinox](https://github.com/patrick-kidger/equinox) module, call `.infer(emissions, method=...)`, and cuthbert handles the numerics.

cuthbert-models is the white chocolate face at the front of the [cuthbert](https://github.com/state-space-models/cuthbert) chocolate-coated sponge cake.


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

## Why

cuthbert is low-level.
You call `build_filter` with callbacks, manage Cholesky decompositions, construct `model_inputs`, and call `cuthbert.filter`.
Each algorithm has its own `build_filter`/`build_smoother` with different callback signatures.

cuthbert-models hides all of that using a bridge layer that converts covariances to Cholesky factors, builds callbacks, and packages raw filter states into typed posteriors.
The user never touches `build_filter`.

Models are `eqx.Module`s, so `jax.grad`, `jax.vmap`, `eqx.partition`, optax, numpyro all work.

## Models

| Model | Infer | Smooth |
|---|---|---|
| `LinearGaussianSSM` | `Kalman`, `Particle` | `Kalman` |
| `NonlinearGaussianSSM` | `EKF`, `UKF`, `Particle` | `EKF`, `UKF` |
| `HMM` | `Forward`, `Particle` | `Forward` |

Methods are frozen dataclasses:

```python
Kalman()                    # sequential
Kalman(parallel=True)       # associative scan
EKF()
UKF()
Forward(parallel=True)
Particle(key=jax.random.key(0), n_particles=500, resampling_fn=resampling)
```

Invalid combinations are blocked by the type system.

### State-dependent emission covariance

dynamax separates `NonlinearGaussianSSM` (emission covariance `R(t)`) from `GeneralizedGaussianSSM` (emission covariance `R(x, t)`, where `x` is the latent state).
This matters for models like heteroscedastic observations where the noise depends on the hidden state.

cuthbert-models doesn't need the split.
`NonlinearGaussianSSM.emission_covariance` takes `(x, t)` by default, subsuming both cases.
If your covariance doesn't depend on state, ignore the first argument: `lambda _x, t: R(t)`.

## Return types

Gaussian models return `GaussianPosterior` (`filtered_means`, `filtered_covariances`).
HMMs return `DiscretePosterior` (`filtered_probs`).
Smoothed variants extend the filtered ones (`smoothed_means`, `smoothed_probs`, etc).

All posteriors carry `marginal_log_likelihood`.

## Outer loop

The model returns a posterior. Parameter estimation is your problem:

- **optax**: differentiate through `.infer()` with `eqx.partition`/`combine`
- **numpyro**: `numpyro.factor("ll", posterior.marginal_log_likelihood)`
- **EM**: smoother E-step + numerical M-step

### Trainable parameters

```python
from cuthbert_models import TrainableWeights, TrainableCovariance

model = LinearGaussianSSM(
    ...
    dynamics_weights=TrainableWeights(F_init),
    dynamics_covariance=TrainableCovariance(Q_init),  # PSD-constrained via numpyro bijectors
    ...
)
```

`TrainableCovariance` requires `numpyro` (`uv sync --extra numpyro`).

## install

Not on PyPI. From source:

```sh
git clone https://github.com/state-space-models/cuthbert-models.git
cd cuthbert-models
uv sync --dev
```

## todo

- **Exogenous inputs / bias terms.** cuthbert's Kalman filter supports bias vectors `c` and `d` in the dynamics and observation equations (`x_t = F @ x_{t-1} + c(t) + noise`). We hardcode them to zero. Adding `dynamics_bias` and `emission_bias` callable fields to `LinearGaussianSSM` would expose this without breaking linearity. For `NonlinearGaussianSSM`, exogenous inputs can already be absorbed into `dynamics_fn`.
- **Continuous-time models.** Drift/diffusion SDE with discretise-then-filter via cd-dynamax or cuthbert.
- **Missing observations.**
- **Forecasting.**

## dev

```sh
uv run ruff check --fix
uv run pytest
```
