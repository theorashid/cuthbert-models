from collections.abc import Callable
from typing import Any

import equinox as eqx
from jaxtyping import Array, Float

from cuthbert_models._types import Posterior

VALID_METHODS = {"forward", "forward_parallel", "particle"}


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
    transition_matrix: Callable[[Any], Float[Array, "states states"]]
    emission_log_likelihood: Callable[[Any, Any], Float[Array, " states"]]

    def infer(
        self,
        emissions: Float[Array, "time obs"],
        method: str = "forward",
        *,
        key: Array | None = None,
        n_particles: int = 100,
        ess_threshold: float = 0.5,
    ) -> Posterior:
        method = _resolve_method(method)

        from cuthbert_models._inference import (  # noqa: PLC0415
            infer_forward,
            infer_particle_hmm,
        )

        if method == "forward":
            return infer_forward(self, emissions, parallel=False)
        if method == "forward_parallel":
            return infer_forward(self, emissions, parallel=True)
        if key is None:
            msg = "method='particle' requires a key argument"
            raise ValueError(msg)
        return infer_particle_hmm(
            self, emissions, key=key,
            n_particles=n_particles, ess_threshold=ess_threshold,
        )


def _resolve_method(method: str) -> str:
    if method not in VALID_METHODS:
        msg = f"Unknown method {method!r}. Valid: {sorted(VALID_METHODS)}"
        raise ValueError(msg)
    return method
