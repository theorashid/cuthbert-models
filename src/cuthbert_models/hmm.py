from collections.abc import Callable

import equinox as eqx
from jaxtyping import Array, Float


class HMM(eqx.Module):

    """
    Hidden Markov model with discrete latent states.

    x_0 ~ Categorical(initial_distribution)
    x_t | x_{t-1} ~ Categorical(transition_matrix(t)[x_{t-1}, :])
    y_t | x_t has log-likelihood given by emission_log_likelihood(y_t, t)

    initial_distribution is a (K,) probability simplex.
    transition_matrix is a callable (t) -> (K, K) row-stochastic matrix.
    emission_log_likelihood is a callable (y, t) -> (K,) log-likelihoods.
    """

    initial_distribution: Float[Array, " states"]
    transition_matrix: Callable[[Float[Array, ""]], Float[Array, "states states"]]
    emission_log_likelihood: Callable[
        [Float[Array, " obs"], Float[Array, ""]], Float[Array, " states"]
    ]
