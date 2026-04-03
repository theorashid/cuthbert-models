# Continuous-time SSMs: design notes

How continuous-time models are implemented here vs dynestyx, and how the dynestyx effect handler system works.

## What we added

Two new model classes, following the same pattern as `LinearGaussianSSM` and `NonlinearGaussianSSM`:

### `LinearContinuousSSM`

```md
dx = A(t) x dt + L(t) dW
y_k = H(t_k) x(t_k) + N(0, R(t_k))
```

The model stores callables `drift_matrix`, `diffusion_coefficient`, `emission_weights`, `emission_covariance`, paralleling the discrete `LinearGaussianSSM` fields (`dynamics_weights`, `dynamics_covariance`, etc.).

With `Filter(Kalman())` and `Discretizer(VanLoan())`, **exactly discretises** each observation interval via the Van Loan (1978) matrix exponential trick, producing time-varying `F(dt)` and `Q(dt)`, then delegates to the existing Kalman filter. No approximation error. Uses `jax.scipy.linalg.expm` -- no new dependencies.

### `NonlinearContinuousSSM`

```md
dx = mu(x, t) dt + L(x, t) dW
y_k = h(x(t_k), t_k) + N(0, R(x(t_k), t_k))
```

With `Filter(EKF(...))` and `Discretizer(EulerMaruyama())`, **Euler-Maruyama discretises** each interval:

- `dynamics_fn(x, t) = x + mu(x, t) * dt`
- `dynamics_covariance(t) = L @ L^T * dt`

then delegates to the existing `NonlinearGaussianSSM` EKF/particle filter. First-order approximation, accurate for small dt.

### Discretization strategies (`_discretize.py`)

Continuous-time models require a `Discretizer` handler, following the same composable pattern as `Filter`:

```python
with Filter(Kalman()), Discretizer(VanLoan()):
    posterior = infer(model, emissions, obs_times=t)

with Filter(EKF(linearization="taylor")), Discretizer(EulerMaruyama()):
    posterior = infer(model, emissions, obs_times=t)
```

- `VanLoan`: exact matrix exponential discretisation (linear models only).
- `EulerMaruyama`: first-order SDE approximation (any model).
- `AdaptiveODE` (commented out): placeholder for a future diffrax-based ODE solver, matching what cd-dynamax does internally.

Validation is in the `Discretizer` and `Filter` handlers (`_handlers.py`):

- `Discretizer(VanLoan())` raises `TypeError` if the model is `NonlinearContinuousSSM`.
- `Discretizer(EulerMaruyama())` converts any continuous model to a discrete `NonlinearGaussianSSM` via `_euler_maruyama_to_discrete`, then `fwd()`s to `Filter`.
- `LinearContinuousSSM` with `Kalman` + `VanLoan` uses exact Van Loan discretisation directly in `Filter` (no `Discretizer` needed, though using one is harmless).
- `LinearContinuousSSM` with `EulerMaruyama` is converted to `NonlinearContinuousSSM` internally (wrapping `diffusion_coefficient(t)` to `(x, t)` signature).

### Design choices

- Continuous-time models pass `obs_times` as a keyword argument to `infer()` and `smooth()`. Discrete models don't need it (time is an integer index).
- Model classes are pure data (equinox Modules, no methods). All dispatch lives in `_handlers.py`.
- Discretisation happens in the `Discretizer` handler or inside `_inference.py` (for Van Loan), invisible to the user. The model stores the continuous-time parameterisation.
- Both models can fall back to particle filtering via `Filter(Particle(...))` + `Discretizer(EulerMaruyama())`.

### Known limitations

- **State-dependent diffusion is not supported.** `NonlinearContinuousSSM.diffusion_coefficient` has signature `(x, t)` for mathematical generality, but the Euler-Maruyama discretisation in `_inference.py` evaluates it at `x=0` (a dummy state) because `NonlinearGaussianSSM.dynamics_covariance` only takes `(t,)`. This is correct for additive noise (the common case). Supporting multiplicative/state-dependent noise would require `dynamics_covariance` to become state-dependent in the discrete model, which would affect all EKF callback factories.

## How dynestyx does the same thing

### The model

Dynestyx has one class `DynamicalModel` that holds everything:

```python
DynamicalModel(
    initial_condition=dist.MultivariateNormal(...),
    state_evolution=ContinuousTimeStateEvolution(
        drift=lambda x, u, t: ...,
        diffusion_coefficient=lambda x, u, t: ...,
    ),
    observation_model=LinearGaussianObservation(H=H, R=R),
)
```

The continuous vs discrete distinction is encoded by the *type* of `state_evolution` (`ContinuousTimeStateEvolution` vs `DiscreteTimeStateEvolution`), not by the model class itself.

### Discretisation

Dynestyx separates discretisation from filtering via the `Discretizer` handler. Euler-Maruyama lives in `dynestyx/discretizers.py:62-67`:

```python
# _EulerMaruyamaDiscreteEvolution.__call__._step
drift = self.cte.total_drift(_x, _u, _t_now)
x_pred_mean = _x + drift * _dt
L = self.cte.diffusion_coefficient(_x, _u, _t_now)
Q = jnp.eye(self.cte.bm_dim)
x_pred_cov = L @ Q @ L.T * _dt
```

This is the same maths we use. The `Q = jnp.eye(bm_dim)` term is because dynestyx factors the diffusion as `L @ sqrt(Q) @ dW` where `Q = I` (identity covariance for the Brownian motion), so `L @ Q @ L^T = L @ L^T`.

For exact linear discretisation, dynestyx delegates to cd-dynamax's internal solver rather than doing the Van Loan trick itself. The model parameters are passed via `dsx_to_cdlgssm_params` (`dynestyx/inference/integrations/cd_dynamax/utils.py:163-195`), which extracts `A`, `L` from the `AffineDrift` and `ContinuousTimeStateEvolution` and hands them to cd-dynamax's `ContDiscreteLinearGaussianSSM.initialize()`.

### Filtering

Once discretised, dynestyx dispatches to a filter backend (cuthbert or cd-dynamax) based on the `filter_config`. For continuous-time models, it typically routes to cd-dynamax's continuous-discrete filters which integrate moment ODEs between observations using diffrax internally.

## How dynestyx's effect handler system works

Dynestyx uses the `effectful` library to implement a pattern from programming language theory called **algebraic effects**. Here's how it works:

### The primitive

```python
# handlers.py
@defop
def _sample_intp(name, dynamics, *, obs_times=None, obs_values=None, ...):
    raise NotHandled()
```

`@defop` declares `_sample_intp` as an **effect operation** -- a function with *no default implementation*. Calling it outside any handler raises `NotHandled`. It's an abstract method, but for free functions rather than classes.

The public `dsx.sample()` validates inputs, then calls `_sample_intp(...)`.

### The handler

A handler is an object that provides an implementation for the effect operation. `Filter` inherits from `ObjectInterpretation` (effectful's base class for object-based handlers) and `HandlesSelf` (a mixin that makes it usable as a context manager):

```python
# filters.py
class Filter(BaseLogFactorAdder):
    filter_config: BaseFilterConfig | None = None

    # This method "handles" the _sample_intp effect
    @implements(_sample_intp)
    def _sample_ds(self, name, dynamics, *, obs_times=None, obs_values=None, ...):
        # Run the filter, add numpyro.factor, return
        ...
```

`@implements(_sample_intp)` says: when `_sample_intp` is called inside my context, run `_sample_ds` instead.

### The context manager

```python
# handlers.py
class HandlesSelf:
    def __enter__(self):
        self._cm = handler(self)      # effectful wraps self as an interpretation
        self._cm.__enter__()           # push onto the handler stack
        return self._cm

    def __exit__(self, exc_type, exc, tb):
        return self._cm.__exit__(...)  # pop from the handler stack
```

`handler(self)` from effectful creates a context manager that intercepts calls to any `@defop` function that `self` has an `@implements` method for. While inside `with Filter(...)`, all calls to `_sample_intp` are routed to `Filter._sample_ds`.

### Handler stacking

Handlers compose by nesting:

```python
with Filter(filter_config=EKFConfig()):
    with Discretizer(discretize=euler_maruyama):
        dsx.sample("f", model, obs_times=t, obs_values=y)
```

When `dsx.sample` calls `_sample_intp`:

1. The innermost handler (`Discretizer`) intercepts first
2. `Discretizer._sample_ds` replaces the continuous-time `state_evolution` with an Euler-Maruyama discrete version
3. It calls `fwd(...)` which re-raises the effect to the next handler up the stack
4. `Filter._sample_ds` intercepts, runs the discrete filter, adds `numpyro.factor`

`fwd(...)` is the key: it means "I've done my part, pass the (possibly modified) call to the next handler." This is how discretisation and filtering are composed without either knowing about the other.

### Why this exists

The effect system lets dynestyx compose operations (discretise, filter, simulate, predict) as independent, stackable layers. Each handler only knows about its own concern.

The cost: the user must understand handler stacking, the `with` syntax is mandatory, and the model can't do anything standalone -- you can't call `model.filter(data)` in a plain script. Every operation requires being inside a handler context.

### What we do (same pattern, simpler)

We now use `effectful` too, but with a simpler design:

```python
from cuthbert_models import Filter, Discretizer, infer, EKF, EulerMaruyama

model = NonlinearContinuousSSM(...)
with Filter(EKF(linearization="taylor")):
    with Discretizer(EulerMaruyama()):
        posterior = infer(model, emissions, obs_times=t)
```

Two `@defop` operations (`infer`, `smooth`) and two handlers (`Filter`, `Discretizer`). Model classes are pure data (equinox Modules with no methods). The existing config objects (`Kalman`, `EKF`, `Particle`, `VanLoan`, `EulerMaruyama`) are passed to the handlers, not to model methods.

Key differences from dynestyx:

- No coupling to numpyro. `Filter` returns a `Posterior` object; the user calls `numpyro.factor` themselves if they want.
- No `dsx.sample()` indirection. `infer(model, emissions)` is the operation.
- `Discretizer` uses `fwd(...)` to pass the discretised model up to `Filter`, same composability pattern as dynestyx.

## Numerical comparison: Van Loan vs cd-dynamax ODE solver

`scripts/compare_dynestyx.py` runs both libraries on the same scalar Ornstein-Uhlenbeck process with identical parameters and data.

### Log-likelihood at true parameters

```md
dynestyx ll:        -38.140659
cuthbert-models ll: -38.098015
difference:         4.26e-02
```

### Why they don't match

The two libraries use fundamentally different discretisation strategies for linear continuous-time models:

- **cuthbert-models**: Van Loan matrix exponential. Exact closed-form discretisation -- `F` and `Q` are computed directly from `expm` of a `2s x 2s` block matrix. The only error is floating-point precision of `jax.scipy.linalg.expm`.

- **dynestyx/cd-dynamax**: Numerical ODE integration via diffrax. `compute_pushforward` in cd-dynamax solves a system of three coupled ODEs (`dA/dt`, `dQ/dt`, `dC/dt`) from `t0` to `t1` using an adaptive ODE solver. This introduces numerical integration error at each timestep, controlled by solver tolerances (`diffeqsolve_dt0=0.01`, `diffeqsolve_max_steps=1000`). The error accumulates over the 100 observation steps.

Both methods are mathematically equivalent for a linear time-invariant system, but the ODE solver is approximate. You could tighten cd-dynamax's tolerances to reduce the gap, but it will never reach machine precision.

### Speed

The Van Loan approach is also faster. The SVI loop ran at ~168 it/s for cuthbert-models vs ~48 it/s for dynestyx (3.5x faster). This makes sense: `expm` of a small block matrix is a single dense operation, while the ODE solver must take many internal sub-steps with adaptive step-size control, function evaluations, and error checking at each. For the nonlinear case (Euler-Maruyama), the two libraries would be closer in speed since both just do simple arithmetic -- but for the linear case, the exact discretisation wins on both accuracy and speed.

## Phase 3: simulation with diffrax (not yet implemented)

Phases 1 and 2 required no new dependencies because they never solve a differential equation numerically:

- **Exact discretisation** uses `jax.scipy.linalg.expm` -- a matrix exponential, not a solver.
- **Euler-Maruyama** is single-step arithmetic: `x + drift * dt` and `L L^T * dt`. No solver loop.

diffrax would be needed for **forward simulation** from the SDE -- sampling actual trajectories, not just filtering given observations. This means:

- **ODE integration** (when `L = 0`): adaptive solvers like `Tsit5` that take many internal sub-steps between observation times, adapting step size where the dynamics change quickly. More accurate than a single Euler step, especially for stiff systems or large `dt`.
- **SDE integration** (when `L != 0`): stochastic solvers like `Heun` with `VirtualBrownianTree` for memory-efficient, consistent Brownian motion generation. The virtual tree generates increments on demand without storing the full path.
- **Multi-sub-step discretisation**: splitting each observation interval into many finer Euler-Maruyama steps rather than one. This would improve filtering accuracy for large `dt` without requiring the exact discretisation (which only works for linear SDEs). diffrax would manage the sub-stepping and adaptive control.

In dynestyx, this is the `Simulator` hierarchy (`dynestyx/simulators.py`):

- `ODESimulator`: wraps `diffrax.diffeqsolve` with `ODETerm`, defaults to `Tsit5`
- `SDESimulator`: wraps `diffrax.diffeqsolve` with `MultiTerm(ODETerm, ControlTerm)` using `VirtualBrownianTree`, defaults to `Heun`
- `DiscreteTimeSimulator`: plain `lax.scan` loop, no solver needed

A `simulate(model, obs_times, key)` function returning sampled states and observations would be the natural addition. diffrax would be an optional dependency (like numpyro is now).
