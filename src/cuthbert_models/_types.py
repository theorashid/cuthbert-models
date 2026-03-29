import equinox as eqx
from jaxtyping import Array, Float


class Posterior(eqx.Module):
    marginal_log_likelihood: Float[Array, ""]
    filtered_means: Float[Array, "time ..."]
    filtered_covariances: Float[Array, "time ..."]


class SmoothedPosterior(eqx.Module):
    marginal_log_likelihood: Float[Array, ""]
    filtered_means: Float[Array, "time ..."]
    filtered_covariances: Float[Array, "time ..."]
    smoothed_means: Float[Array, "time ..."]
    smoothed_covariances: Float[Array, "time ..."]
