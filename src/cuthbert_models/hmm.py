from collections.abc import Callable
from typing import Any

import equinox as eqx
from jaxtyping import Array, Float

from cuthbert_models._methods import Forward, Particle
from cuthbert_models._types import Posterior, SmoothedPosterior

Method = Forward | Particle
_DEFAULT_METHOD = Forward()


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
        method: Method = _DEFAULT_METHOD,
    ) -> Posterior:
        from cuthbert_models._inference import (  # noqa: PLC0415
            infer_forward,
            infer_particle_hmm,
        )

        if isinstance(method, Forward):
            return infer_forward(self, emissions, parallel=method.parallel)
        return infer_particle_hmm(
            self, emissions, key=method.key,
            n_particles=method.n_particles, ess_threshold=method.ess_threshold,
        )

    def smooth(
        self,
        emissions: Float[Array, "time obs"],
        method: Forward = _DEFAULT_METHOD,
    ) -> SmoothedPosterior:
        from cuthbert_models._inference import smooth_forward  # noqa: PLC0415

        return smooth_forward(self, emissions, parallel=method.parallel)
