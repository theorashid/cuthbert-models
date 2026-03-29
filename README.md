# cuthbert-models

The white chocolate face on [cuthbert](https://github.com/state-space-models/cuthbert)'s chocolate cake backend.

`cuthbert-models` is a model specification layer for state space models.
You define the generative model as an [Equinox](https://github.com/patrick-kidger/equinox) module, call `.infer(emissions, method=...)`, and cuthbert handles the numerics.

```python
from cuthbert_models import LinearGaussianSSM

model = LinearGaussianSSM(
    initial_mean=m0,
    initial_covariance=P0,
    dynamics_weights=lambda t: F,
    dynamics_covariance=lambda t: Q,
    emission_weights=lambda t: H,
    emission_covariance=lambda t: R,
)
posterior = model.infer(emissions, method="kalman")
```

The model defines the generative process.
The method chooses the inference algorithm.
cuthbert does the work.

## Models

Each model is an `eqx.Module` with `.infer()`.
All parameter fields are callables `(t) -> Array`, matching cuthbert's callback contract.

| Model | Structure | Methods |
|---|---|---|
| `LinearGaussianSSM` | Linear dynamics, Gaussian noise | `kalman`, `kalman_parallel`, `ekf`, `ukf`, `particle` |
| `NonlinearGaussianSSM` | Nonlinear dynamics, Gaussian noise | `ekf`, `ukf`, `particle` |
| `HMM` | Discrete latent states | `forward`, `forward_parallel`, `particle` |

Linear models accept all methods (Kalman is exact, the rest work but are overkill).
Nonlinear models reject `kalman` (not exact).
Particle filtering works on anything.

## Inner loop, outer loop

The inner loop (filtering) is handled by cuthbert.
The outer loop (parameter estimation) is your choice:

- **optax**: MLE via `eqx.partition`/`combine`, differentiate through `.infer()`
- **numpyro**: `numpyro.factor("ll", model.infer(y).marginal_log_likelihood)`
- **blackjax**: same scalar log-likelihood

The model doesn't care which outer loop you use.

## Install

```sh
pip install cuthbert-models
```

## Dev

```sh
uv sync --extra test
uv run ruff check --fix
uv run pytest
```
