# Effectful handlers for inference and discretisation.
#
# Same effectful patterns as dynestyx (effectful.ops.syntax,
# ObjectInterpretation, HandlesSelf mixin, fwd for handler chaining).
#
# Differences from dynestyx:
# - Two separate @defop operations (infer, smooth) instead of one
#   (_sample_intp) that multiplexes filter/simulate/predict.
# - Filter does not call numpyro.factor; it returns a Posterior and
#   the user calls numpyro.factor themselves.
# - Discretizer takes a config object (VanLoan(), EulerMaruyama())
#   rather than a function (euler_maruyama).
# - Filter is terminal (returns result) rather than calling fwd()
#   to pass filtered_dists to a Simulator/Predictor above.
#
# Usage:
#   with Filter(Kalman()):
#       posterior = infer(model, emissions)
#
#   with Filter(Kalman()), Discretizer(VanLoan()):
#       posterior = infer(model, emissions, obs_times=t)

from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, defop, implements
from effectful.ops.types import NotHandled
from jaxtyping import Array, Float

from cuthbert_models._discretize import (
    EulerMaruyama,
    VanLoan,
    euler_maruyama_to_discrete,
)
from cuthbert_models._methods import EKF, Forward, Kalman, Particle
from cuthbert_models._types import (
    DiscretePosterior,
    DiscreteSmoothedPosterior,
    GaussianPosterior,
    GaussianSmoothedPosterior,
    Posterior,
)

# ---------------------------------------------------------------------------
# Effect operations
# ---------------------------------------------------------------------------


@defop
def infer(model, emissions, *, obs_times=None) -> Posterior:  # noqa: ARG001
    raise NotHandled


@defop
def smooth(model, emissions, *, obs_times=None) -> Posterior:  # noqa: ARG001
    raise NotHandled


# ---------------------------------------------------------------------------
# HandlesSelf mixin (identical to dynestyx.handlers.HandlesSelf)
# ---------------------------------------------------------------------------


class _HandlesSelf:
    _cm: object = None

    def __enter__(self):
        self._cm = handler(self)
        self._cm.__enter__()  # type: ignore[union-attribute]
        return self._cm

    def __exit__(self, exc_type, exc, tb):
        return self._cm.__exit__(exc_type, exc, tb)  # type: ignore[union-attribute]


# ---------------------------------------------------------------------------
# Filter handler
# ---------------------------------------------------------------------------


class Filter(ObjectInterpretation, _HandlesSelf):

    def __init__(self, method):
        super().__init__()
        self.method = method

    @implements(infer)
    def _infer(self, model, emissions, *, obs_times=None):
        from cuthbert_models.hmm import HMM  # noqa: PLC0415
        from cuthbert_models.linear_continuous import (  # noqa: PLC0415
            LinearContinuousSSM,
        )
        from cuthbert_models.linear_gaussian import LinearGaussianSSM  # noqa: PLC0415
        from cuthbert_models.nonlinear_continuous import (  # noqa: PLC0415
            NonlinearContinuousSSM,
        )
        from cuthbert_models.nonlinear_gaussian import (  # noqa: PLC0415
            NonlinearGaussianSSM,
        )

        method = self.method

        if isinstance(model, LinearContinuousSSM | NonlinearContinuousSSM):
            if obs_times is None:
                msg = "obs_times is required for continuous-time models."
                raise ValueError(msg)
            return _infer_continuous(model, emissions, obs_times, method)

        if isinstance(model, LinearGaussianSSM):
            return _infer_linear_gaussian(model, emissions, method)

        if isinstance(model, NonlinearGaussianSSM):
            return _infer_nonlinear_gaussian(model, emissions, method)

        if isinstance(model, HMM):
            return _infer_hmm(model, emissions, method)

        msg = f"Unsupported model type: {type(model)}"
        raise TypeError(msg)

    @implements(smooth)
    def _smooth(self, model, emissions, *, obs_times=None):
        from cuthbert_models.hmm import HMM  # noqa: PLC0415
        from cuthbert_models.linear_continuous import (  # noqa: PLC0415
            LinearContinuousSSM,
        )
        from cuthbert_models.linear_gaussian import LinearGaussianSSM  # noqa: PLC0415
        from cuthbert_models.nonlinear_continuous import (  # noqa: PLC0415
            NonlinearContinuousSSM,
        )
        from cuthbert_models.nonlinear_gaussian import (  # noqa: PLC0415
            NonlinearGaussianSSM,
        )

        method = self.method

        if isinstance(model, LinearContinuousSSM | NonlinearContinuousSSM):
            if obs_times is None:
                msg = "obs_times is required for continuous-time models."
                raise ValueError(msg)
            return _smooth_continuous(model, emissions, obs_times, method)

        if isinstance(model, LinearGaussianSSM):
            return _smooth_linear_gaussian(model, emissions, method)

        if isinstance(model, NonlinearGaussianSSM):
            return _smooth_nonlinear_gaussian(model, emissions, method)

        if isinstance(model, HMM):
            return _smooth_hmm(model, emissions, method)

        msg = f"Unsupported model type: {type(model)}"
        raise TypeError(msg)


# ---------------------------------------------------------------------------
# Discretizer handler (uses fwd to chain to Filter, same as dynestyx)
# ---------------------------------------------------------------------------


class Discretizer(ObjectInterpretation, _HandlesSelf):

    def __init__(self, discretization):
        super().__init__()
        self.discretization = discretization

    @implements(infer)
    def _infer(self, model, emissions, *, obs_times=None):
        model, obs_times = self._discretize(model, obs_times)
        return fwd(model, emissions, obs_times=obs_times)

    @implements(smooth)
    def _smooth(self, model, emissions, *, obs_times=None):
        model, obs_times = self._discretize(model, obs_times)
        return fwd(model, emissions, obs_times=obs_times)

    def _discretize(self, model, obs_times):
        from cuthbert_models.linear_continuous import (  # noqa: PLC0415
            LinearContinuousSSM,
        )
        from cuthbert_models.nonlinear_continuous import (  # noqa: PLC0415
            NonlinearContinuousSSM,
        )

        if not isinstance(model, LinearContinuousSSM | NonlinearContinuousSSM):
            return model, obs_times

        if obs_times is None:
            msg = "obs_times is required for continuous-time models."
            raise ValueError(msg)

        disc = self.discretization

        if isinstance(disc, VanLoan):
            if isinstance(model, NonlinearContinuousSSM):
                msg = (
                    "VanLoan discretization requires a linear model. "
                    "Use EulerMaruyama() for NonlinearContinuousSSM."
                )
                raise TypeError(msg)
            return model, obs_times

        if isinstance(disc, EulerMaruyama):
            return euler_maruyama_to_discrete(model, obs_times), None

        msg = f"Unsupported discretization: {type(disc)}"
        raise TypeError(msg)


# ---------------------------------------------------------------------------
# NumpyroTrace handler (registers numpyro.factor and deterministic sites)
# ---------------------------------------------------------------------------


class NumpyroTrace(ObjectInterpretation, _HandlesSelf):

    def __init__(self, name="ssm", *, deterministic=True):
        super().__init__()
        self.name = name
        self.deterministic = deterministic

    @implements(infer)
    def _infer(self, model, emissions, *, obs_times=None):
        posterior = fwd(model, emissions, obs_times=obs_times)
        return self._register(posterior)

    @implements(smooth)
    def _smooth(self, model, emissions, *, obs_times=None):
        posterior = fwd(model, emissions, obs_times=obs_times)
        return self._register(posterior)

    def _register(self, posterior):
        import numpyro  # noqa: PLC0415

        numpyro.factor(
            f"{self.name}_log_likelihood", posterior.marginal_log_likelihood,
        )
        if not self.deterministic:
            return posterior

        numpyro.deterministic(
            f"{self.name}_marginal_log_likelihood",
            posterior.marginal_log_likelihood,
        )
        if isinstance(posterior, GaussianSmoothedPosterior):
            numpyro.deterministic(
                f"{self.name}_filtered_means", posterior.filtered_means,
            )
            numpyro.deterministic(
                f"{self.name}_filtered_covariances",
                posterior.filtered_covariances,
            )
            numpyro.deterministic(
                f"{self.name}_smoothed_means", posterior.smoothed_means,
            )
            numpyro.deterministic(
                f"{self.name}_smoothed_covariances",
                posterior.smoothed_covariances,
            )
        elif isinstance(posterior, GaussianPosterior):
            numpyro.deterministic(
                f"{self.name}_filtered_means", posterior.filtered_means,
            )
            numpyro.deterministic(
                f"{self.name}_filtered_covariances",
                posterior.filtered_covariances,
            )
        elif isinstance(posterior, DiscreteSmoothedPosterior):
            numpyro.deterministic(
                f"{self.name}_filtered_probs", posterior.filtered_probs,
            )
            numpyro.deterministic(
                f"{self.name}_smoothed_probs", posterior.smoothed_probs,
            )
        elif isinstance(posterior, DiscretePosterior):
            numpyro.deterministic(
                f"{self.name}_filtered_probs", posterior.filtered_probs,
            )
        return posterior


# ---------------------------------------------------------------------------
# Dispatch helpers (delegate to _inference.py)
# ---------------------------------------------------------------------------


def _infer_linear_gaussian(
    model, emissions, method,
) -> GaussianPosterior:
    from cuthbert_models._inference import (  # noqa: PLC0415
        infer_kalman,
        infer_particle_gaussian,
    )

    if isinstance(method, Kalman):
        return infer_kalman(model, emissions, parallel=method.parallel)

    if isinstance(method, Particle):
        nonlinear = _linear_to_nonlinear(model)
        return infer_particle_gaussian(
            nonlinear, emissions, key=method.key,
            n_particles=method.n_particles, resampling_fn=method.resampling_fn,
        )

    msg = f"LinearGaussianSSM does not support method {type(method).__name__}."
    raise TypeError(msg)


def _smooth_linear_gaussian(
    model, emissions, method,
) -> GaussianSmoothedPosterior:
    from cuthbert_models._inference import smooth_kalman  # noqa: PLC0415

    if not isinstance(method, Kalman):
        msg = "LinearGaussianSSM smoothing requires Kalman()."
        raise TypeError(msg)
    return smooth_kalman(model, emissions, parallel=method.parallel)


def _infer_nonlinear_gaussian(
    model, emissions, method,
) -> GaussianPosterior:
    from cuthbert_models._inference import (  # noqa: PLC0415
        infer_ekf,
        infer_ekf_moments,
        infer_particle_gaussian,
    )

    if isinstance(method, EKF):
        if method.linearization == "moments":
            return infer_ekf_moments(model, emissions)
        return infer_ekf(model, emissions)

    if isinstance(method, Particle):
        return infer_particle_gaussian(
            model, emissions, key=method.key,
            n_particles=method.n_particles, resampling_fn=method.resampling_fn,
        )

    msg = f"NonlinearGaussianSSM does not support method {type(method).__name__}."
    raise TypeError(msg)


def _smooth_nonlinear_gaussian(
    model, emissions, method,
) -> GaussianSmoothedPosterior:
    from cuthbert_models._inference import (  # noqa: PLC0415
        smooth_ekf,
        smooth_ekf_moments,
    )

    if not isinstance(method, EKF):
        msg = "NonlinearGaussianSSM smoothing requires EKF()."
        raise TypeError(msg)
    if method.linearization == "moments":
        return smooth_ekf_moments(model, emissions)
    return smooth_ekf(model, emissions)


def _infer_hmm(
    model, emissions, method,
) -> DiscretePosterior:
    from cuthbert_models._inference import (  # noqa: PLC0415
        infer_forward,
        infer_particle_hmm,
    )

    if isinstance(method, Forward):
        return infer_forward(model, emissions, parallel=method.parallel)

    if isinstance(method, Particle):
        return infer_particle_hmm(
            model, emissions, key=method.key,
            n_particles=method.n_particles, resampling_fn=method.resampling_fn,
        )

    msg = f"HMM does not support method {type(method).__name__}."
    raise TypeError(msg)


def _smooth_hmm(
    model, emissions, method,
) -> DiscreteSmoothedPosterior:
    from cuthbert_models._inference import smooth_forward  # noqa: PLC0415

    if not isinstance(method, Forward):
        msg = "HMM smoothing requires Forward()."
        raise TypeError(msg)
    return smooth_forward(model, emissions, parallel=method.parallel)


# ---------------------------------------------------------------------------
# Continuous-time dispatch
# ---------------------------------------------------------------------------


def _infer_continuous(
    model,
    emissions: Float[Array, "time obs"],
    obs_times: Float[Array, " time"],
    method,
) -> GaussianPosterior:
    from cuthbert_models._inference import infer_linear_continuous  # noqa: PLC0415
    from cuthbert_models.linear_continuous import (  # noqa: PLC0415
        LinearContinuousSSM,
    )

    if isinstance(model, LinearContinuousSSM) and isinstance(method, Kalman):
        return infer_linear_continuous(
            model, obs_times, emissions, parallel=method.parallel,
        )

    msg = (
        "Continuous-time model with this method requires a Discretizer. "
        "Wrap with: with Discretizer(EulerMaruyama()): ..."
    )
    raise TypeError(msg)


def _smooth_continuous(
    model,
    emissions: Float[Array, "time obs"],
    obs_times: Float[Array, " time"],
    method,
) -> GaussianSmoothedPosterior:
    from cuthbert_models._inference import smooth_linear_continuous  # noqa: PLC0415
    from cuthbert_models.linear_continuous import (  # noqa: PLC0415
        LinearContinuousSSM,
    )

    if isinstance(model, LinearContinuousSSM) and isinstance(method, Kalman):
        return smooth_linear_continuous(
            model, obs_times, emissions, parallel=method.parallel,
        )

    msg = (
        "Continuous-time model smoothing with this method requires a Discretizer. "
        "Wrap with: with Discretizer(EulerMaruyama()): ..."
    )
    raise TypeError(msg)


# ---------------------------------------------------------------------------
# Model conversions
# ---------------------------------------------------------------------------


def _linear_to_nonlinear(model):
    from cuthbert_models.nonlinear_gaussian import (  # noqa: PLC0415
        NonlinearGaussianSSM,
    )

    return NonlinearGaussianSSM(
        initial_mean=model.initial_mean,
        initial_covariance=model.initial_covariance,
        dynamics_fn=lambda x, t: model.dynamics_weights(t) @ x,
        dynamics_covariance=model.dynamics_covariance,
        emission_fn=lambda x, t: model.emission_weights(t) @ x,
        emission_covariance=lambda _x, t: model.emission_covariance(t),
    )


def _linear_continuous_to_nonlinear(model):
    from cuthbert_models.nonlinear_continuous import (  # noqa: PLC0415
        NonlinearContinuousSSM,
    )

    _diff = model.diffusion_coefficient
    return NonlinearContinuousSSM(
        initial_mean=model.initial_mean,
        initial_covariance=model.initial_covariance,
        drift=lambda x, t: model.drift_matrix(t) @ x,
        diffusion_coefficient=lambda _x, t: _diff(t),
        emission_fn=lambda x, t: model.emission_weights(t) @ x,
        emission_covariance=lambda _x, t: model.emission_covariance(t),
    )
