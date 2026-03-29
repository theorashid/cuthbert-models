import equinox as eqx
from jaxtyping import Array, Float


class Posterior(eqx.Module):
    marginal_log_likelihood: Float[Array, ""]


class GaussianPosterior(Posterior):
    filtered_means: Float[Array, "time state"]
    filtered_covariances: Float[Array, "time state state"]


class DiscretePosterior(Posterior):
    filtered_probs: Float[Array, "time state"]


class GaussianSmoothedPosterior(GaussianPosterior):
    smoothed_means: Float[Array, "time state"]
    smoothed_covariances: Float[Array, "time state state"]


class DiscreteSmoothedPosterior(DiscretePosterior):
    smoothed_probs: Float[Array, "time state"]
