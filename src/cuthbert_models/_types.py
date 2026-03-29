import abc

import equinox as eqx
from jaxtyping import Array, Float


class Posterior(eqx.Module):
    marginal_log_likelihood: Float[Array, ""]

    @abc.abstractmethod
    def _posterior_type(self) -> str: ...


class GaussianPosterior(Posterior):
    filtered_means: Float[Array, "time state"]
    filtered_covariances: Float[Array, "time state state"]

    def _posterior_type(self) -> str:
        return "gaussian"


class DiscretePosterior(Posterior):
    filtered_probs: Float[Array, "time state"]

    def _posterior_type(self) -> str:
        return "discrete"


class SmoothedPosterior(Posterior):
    @abc.abstractmethod
    def _smoothed_posterior_type(self) -> str: ...


class GaussianSmoothedPosterior(SmoothedPosterior):
    filtered_means: Float[Array, "time state"]
    filtered_covariances: Float[Array, "time state state"]
    smoothed_means: Float[Array, "time state"]
    smoothed_covariances: Float[Array, "time state state"]

    def _posterior_type(self) -> str:
        return "gaussian"

    def _smoothed_posterior_type(self) -> str:
        return "gaussian"


class DiscreteSmoothedPosterior(SmoothedPosterior):
    filtered_probs: Float[Array, "time state"]
    smoothed_probs: Float[Array, "time state"]

    def _posterior_type(self) -> str:
        return "discrete"

    def _smoothed_posterior_type(self) -> str:
        return "discrete"
