# Bridge from model objects to cuthbert callbacks.
#
# Each public function converts a model's callable fields into the callback
# protocols that cuthbert expects, then runs the filter/smoother.
#
# cuthbert's native parameterisation is Cholesky factors. We decompose
# the user's full covariance matrices here. If the model stored Cholesky
# factors directly, these decompositions could be skipped.

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as jr
from cuthbert import filter as cuthbert_filter
from cuthbert import smoother as cuthbert_smoother
from cuthbert.discrete import filter as discrete_filter
from cuthbert.discrete import smoother as discrete_smoother_mod
from cuthbert.gaussian import kalman
from cuthbert.gaussian.moments import filter as moments_filter
from cuthbert.gaussian.moments import smoother as moments_smoother_mod
from cuthbert.gaussian.taylor import filter as taylor_filter
from cuthbert.gaussian.taylor import smoother as taylor_smoother_mod
from cuthbert.smc import particle_filter as pf
from cuthbertlib.resampling.systematic import resampling as systematic_resampling
from jaxtyping import Array, Float, Key

from cuthbert_models._types import (
    DiscretePosterior,
    DiscreteSmoothedPosterior,
    GaussianPosterior,
    GaussianSmoothedPosterior,
)

if TYPE_CHECKING:
    from cuthbert_models.hmm import HMM
    from cuthbert_models.linear_gaussian import LinearGaussianSSM
    from cuthbert_models.nonlinear_gaussian import NonlinearGaussianSSM


# ---------------------------------------------------------------------------
# Shared posterior constructors
# ---------------------------------------------------------------------------


def _gaussian_posterior(states) -> GaussianPosterior:
    filtered_means = states.mean[1:]
    filtered_chol_covs = states.chol_cov[1:]
    filtered_covariances = (
        filtered_chol_covs @ jnp.matrix_transpose(filtered_chol_covs)
    )
    return GaussianPosterior(
        marginal_log_likelihood=states.log_normalizing_constant[-1],
        filtered_means=filtered_means,
        filtered_covariances=filtered_covariances,
    )


def _gaussian_smoothed_posterior(
    filter_states, smoother_states,
) -> GaussianSmoothedPosterior:
    filtered = _gaussian_posterior(filter_states)
    smoothed_means = smoother_states.mean[1:]
    smoothed_chol_covs = smoother_states.chol_cov[1:]
    smoothed_covariances = (
        smoothed_chol_covs @ jnp.matrix_transpose(smoothed_chol_covs)
    )
    return GaussianSmoothedPosterior(
        marginal_log_likelihood=filtered.marginal_log_likelihood,
        filtered_means=filtered.filtered_means,
        filtered_covariances=filtered.filtered_covariances,
        smoothed_means=smoothed_means,
        smoothed_covariances=smoothed_covariances,
    )


# ---------------------------------------------------------------------------
# Kalman callback factories
# ---------------------------------------------------------------------------


def _kalman_callbacks(
    model: LinearGaussianSSM,
    emissions: Float[Array, "time obs"],
):
    dtype = model.initial_mean.dtype
    state_dim = model.initial_mean.shape[0]
    obs_dim = emissions.shape[-1]

    chol_P0 = jnp.linalg.cholesky(model.initial_covariance)
    c = jnp.zeros(state_dim, dtype=dtype)
    d = jnp.zeros(obs_dim, dtype=dtype)

    def get_init_params(_mi):
        return model.initial_mean, chol_P0

    def get_dynamics_params(mi):
        t = jnp.array(mi - 1, dtype=dtype)
        F_t = model.dynamics_weights(t)
        Q_t = model.dynamics_covariance(t)
        return F_t, c, jnp.linalg.cholesky(Q_t)

    def get_observation_params(mi):
        t = jnp.array(mi - 1, dtype=dtype)
        t_idx = jnp.array(mi - 1, dtype=jnp.int32)
        H_t = model.emission_weights(t)
        R_t = model.emission_covariance(t)
        return H_t, d, jnp.linalg.cholesky(R_t), emissions[t_idx]

    return get_init_params, get_dynamics_params, get_observation_params


def infer_kalman(
    model: LinearGaussianSSM,
    emissions: Float[Array, "time obs"],
    *,
    parallel: bool = False,
) -> GaussianPosterior:
    get_init, get_dyn, get_obs = _kalman_callbacks(model, emissions)
    filter_obj = kalman.build_filter(get_init, get_dyn, get_obs)
    model_inputs = jnp.arange(emissions.shape[0] + 1)
    states = cuthbert_filter(filter_obj, model_inputs, parallel=parallel)
    return _gaussian_posterior(states)


def smooth_kalman(
    model: LinearGaussianSSM,
    emissions: Float[Array, "time obs"],
    *,
    parallel: bool = False,
) -> GaussianSmoothedPosterior:
    get_init, get_dyn, get_obs = _kalman_callbacks(model, emissions)
    filter_obj = kalman.build_filter(get_init, get_dyn, get_obs)
    smoother_obj = kalman.build_smoother(get_dyn)
    model_inputs = jnp.arange(emissions.shape[0] + 1)
    filter_states = cuthbert_filter(filter_obj, model_inputs, parallel=parallel)
    smoother_states = cuthbert_smoother(
        smoother_obj, filter_states, parallel=parallel,
    )
    return _gaussian_smoothed_posterior(filter_states, smoother_states)


# ---------------------------------------------------------------------------
# EKF (Taylor) callback factories
# ---------------------------------------------------------------------------


def _ekf_callbacks(
    model: NonlinearGaussianSSM,
    emissions: Float[Array, "time obs"],
):
    dtype = model.initial_mean.dtype
    state_dim = model.initial_mean.shape[0]
    m0 = model.initial_mean
    P0_inv = jnp.linalg.solve(
        model.initial_covariance, jnp.eye(state_dim, dtype=dtype),
    )

    def get_init_log_density(_mi):
        def init_log_density(x):
            diff = x - m0
            return -0.5 * diff @ P0_inv @ diff
        return init_log_density, m0

    def get_dynamics_log_density(state, mi):
        t = jnp.array(mi - 1, dtype=dtype)
        x_prev_lin = state.mean
        Q_t = model.dynamics_covariance(t)
        Q_t_inv = jnp.linalg.solve(Q_t, jnp.eye(state_dim, dtype=dtype))

        def dynamics_log_density(x_prev, x):
            predicted = model.dynamics_fn(x_prev, t)
            diff = x - predicted
            return -0.5 * diff @ Q_t_inv @ diff

        x_lin = model.dynamics_fn(x_prev_lin, t)
        return dynamics_log_density, x_prev_lin, x_lin

    def get_observation_func(state, mi):
        t = jnp.array(mi - 1, dtype=dtype)
        t_idx = jnp.array(mi - 1, dtype=jnp.int32)
        y = emissions[t_idx]
        x_lin = state.mean

        def obs_log_density(x, y_obs):
            predicted = model.emission_fn(x, t)
            R_t = model.emission_covariance(x, t)
            obs_dim = R_t.shape[0]
            R_t_inv = jnp.linalg.solve(R_t, jnp.eye(obs_dim, dtype=dtype))
            diff = y_obs - predicted
            return -0.5 * diff @ R_t_inv @ diff

        return obs_log_density, x_lin, y

    return get_init_log_density, get_dynamics_log_density, get_observation_func


def infer_ekf(
    model: NonlinearGaussianSSM,
    emissions: Float[Array, "time obs"],
) -> GaussianPosterior:
    get_init, get_dyn, get_obs = _ekf_callbacks(model, emissions)
    filter_obj = taylor_filter.build_filter(
        get_init, get_dyn, get_obs, associative=False,
    )
    model_inputs = jnp.arange(emissions.shape[0] + 1)
    states = cuthbert_filter(filter_obj, model_inputs, parallel=False)
    return _gaussian_posterior(states)


def smooth_ekf(
    model: NonlinearGaussianSSM,
    emissions: Float[Array, "time obs"],
) -> GaussianSmoothedPosterior:
    get_init, get_dyn, get_obs = _ekf_callbacks(model, emissions)
    filter_obj = taylor_filter.build_filter(
        get_init, get_dyn, get_obs, associative=False,
    )
    smoother_obj = taylor_smoother_mod.build_smoother(get_dyn)
    model_inputs = jnp.arange(emissions.shape[0] + 1)
    filter_states = cuthbert_filter(filter_obj, model_inputs, parallel=False)
    smoother_states = cuthbert_smoother(
        smoother_obj, filter_states, parallel=False,
    )
    return _gaussian_smoothed_posterior(filter_states, smoother_states)


# ---------------------------------------------------------------------------
# UKF (Moments) callback factories
# ---------------------------------------------------------------------------


def _ukf_callbacks(
    model: NonlinearGaussianSSM,
    emissions: Float[Array, "time obs"],
):
    dtype = model.initial_mean.dtype
    m0 = model.initial_mean
    chol_P0 = jnp.linalg.cholesky(model.initial_covariance)

    def get_init_params(_mi):
        return m0, chol_P0

    def get_dynamics_moments(state, mi):
        t = jnp.array(mi - 1, dtype=dtype)
        Q_t = model.dynamics_covariance(t)
        chol_Q_t = jnp.linalg.cholesky(Q_t)

        def mean_and_chol_cov(x_prev):
            return model.dynamics_fn(x_prev, t), chol_Q_t

        x_lin = state.mean
        return mean_and_chol_cov, x_lin

    def get_observation_moments(state, mi):
        t = jnp.array(mi - 1, dtype=dtype)
        t_idx = jnp.array(mi - 1, dtype=jnp.int32)

        def mean_and_chol_cov(x):
            R_t = model.emission_covariance(x, t)
            return model.emission_fn(x, t), jnp.linalg.cholesky(R_t)

        x_lin = state.mean
        y = emissions[t_idx]
        return mean_and_chol_cov, x_lin, y

    return get_init_params, get_dynamics_moments, get_observation_moments


def infer_ukf(
    model: NonlinearGaussianSSM,
    emissions: Float[Array, "time obs"],
) -> GaussianPosterior:
    get_init, get_dyn, get_obs = _ukf_callbacks(model, emissions)
    filter_obj = moments_filter.build_filter(
        get_init, get_dyn, get_obs, associative=False,
    )
    model_inputs = jnp.arange(emissions.shape[0] + 1)
    states = cuthbert_filter(filter_obj, model_inputs, parallel=False)
    return _gaussian_posterior(states)


def smooth_ukf(
    model: NonlinearGaussianSSM,
    emissions: Float[Array, "time obs"],
) -> GaussianSmoothedPosterior:
    get_init, get_dyn, get_obs = _ukf_callbacks(model, emissions)
    filter_obj = moments_filter.build_filter(
        get_init, get_dyn, get_obs, associative=False,
    )
    smoother_obj = moments_smoother_mod.build_smoother(get_dyn)
    model_inputs = jnp.arange(emissions.shape[0] + 1)
    filter_states = cuthbert_filter(filter_obj, model_inputs, parallel=False)
    smoother_states = cuthbert_smoother(
        smoother_obj, filter_states, parallel=False,
    )
    return _gaussian_smoothed_posterior(filter_states, smoother_states)


# ---------------------------------------------------------------------------
# HMM (Discrete) callback factories
# ---------------------------------------------------------------------------


def _forward_callbacks(
    model: HMM,
    emissions: Float[Array, "time ..."],
):
    dtype = model.initial_distribution.dtype

    def get_init_dist(_mi):
        return model.initial_distribution

    def get_transition_matrix(mi):
        t = jnp.array(mi - 1, dtype=dtype)
        return model.transition_matrix(t)

    def get_obs_log_likelihoods(mi):
        t = jnp.array(mi - 1, dtype=dtype)
        t_idx = jnp.array(mi - 1, dtype=jnp.int32)
        return model.emission_log_likelihood(emissions[t_idx], t)

    return get_init_dist, get_transition_matrix, get_obs_log_likelihoods


def infer_forward(
    model: HMM,
    emissions: Float[Array, "time ..."],
    *,
    parallel: bool = False,
) -> DiscretePosterior:
    get_init, get_trans, get_obs = _forward_callbacks(model, emissions)
    filter_obj = discrete_filter.build_filter(get_init, get_trans, get_obs)
    model_inputs = jnp.arange(emissions.shape[0] + 1)
    states = cuthbert_filter(filter_obj, model_inputs, parallel=parallel)
    return DiscretePosterior(
        marginal_log_likelihood=states.log_normalizing_constant[-1],
        filtered_probs=states.dist[1:],
    )


def smooth_forward(
    model: HMM,
    emissions: Float[Array, "time ..."],
    *,
    parallel: bool = False,
) -> DiscreteSmoothedPosterior:
    get_init, get_trans, get_obs = _forward_callbacks(model, emissions)
    filter_obj = discrete_filter.build_filter(get_init, get_trans, get_obs)
    smoother_obj = discrete_smoother_mod.build_smoother(get_trans)
    model_inputs = jnp.arange(emissions.shape[0] + 1)
    filter_states = cuthbert_filter(filter_obj, model_inputs, parallel=parallel)
    smoother_states = cuthbert_smoother(
        smoother_obj, filter_states, parallel=parallel,
    )
    return DiscreteSmoothedPosterior(
        marginal_log_likelihood=filter_states.log_normalizing_constant[-1],
        filtered_probs=filter_states.dist[1:],
        smoothed_probs=smoother_states.dist[1:],
    )


# ---------------------------------------------------------------------------
# Particle filter: bootstrap PF via cuthbert.smc
# ---------------------------------------------------------------------------


def infer_particle_gaussian(
    model: NonlinearGaussianSSM,
    emissions: Float[Array, "time obs"],
    *,
    key: Key[Array, ""],
    n_particles: int = 100,
    ess_threshold: float = 0.5,
) -> GaussianPosterior:
    dtype = model.initial_mean.dtype
    state_dim = model.initial_mean.shape[0]
    m0 = model.initial_mean
    chol_P0 = jnp.linalg.cholesky(model.initial_covariance)

    def init_sample(key, _mi):
        return m0 + chol_P0 @ jr.normal(key, (state_dim,), dtype=dtype)

    def propagate_sample(key, state, mi):
        t = jnp.array(mi - 1, dtype=dtype)
        mean = model.dynamics_fn(state, t)
        Q_t = model.dynamics_covariance(t)
        chol_Q_t = jnp.linalg.cholesky(Q_t)
        return mean + chol_Q_t @ jr.normal(key, (state_dim,), dtype=dtype)

    def log_potential(_state_prev, state, mi):
        t = jnp.array(mi - 1, dtype=dtype)
        t_idx = jnp.array(mi - 1, dtype=jnp.int32)
        y = emissions[t_idx]
        predicted = model.emission_fn(state, t)
        R_t = model.emission_covariance(state, t)
        obs_dim = R_t.shape[0]
        diff = y - predicted
        chol_R = jnp.linalg.cholesky(R_t)
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(chol_R)))
        solve = jnp.linalg.solve(R_t, diff)
        return -0.5 * (obs_dim * jnp.log(2.0 * jnp.pi) + log_det + diff @ solve)

    filter_obj = pf.build_filter(
        init_sample,
        propagate_sample,
        log_potential,
        n_filter_particles=n_particles,
        resampling_fn=systematic_resampling,
        ess_threshold=ess_threshold,
    )
    model_inputs = jnp.arange(emissions.shape[0] + 1)
    states = cuthbert_filter(filter_obj, model_inputs, parallel=False, key=key)

    log_w = states.log_weights[1:]
    w = jax.nn.softmax(log_w, axis=-1)
    particles = states.particles[1:]
    filtered_means = jnp.einsum("tn,tn...->t...", w, particles)

    return GaussianPosterior(
        marginal_log_likelihood=states.log_normalizing_constant[-1],
        filtered_means=filtered_means,
        filtered_covariances=jnp.zeros_like(
            jnp.broadcast_to(
                jnp.eye(state_dim), (emissions.shape[0], state_dim, state_dim),
            ),
        ),
    )


def infer_particle_hmm(
    model: HMM,
    emissions: Float[Array, "time ..."],
    *,
    key: Key[Array, ""],
    n_particles: int = 100,
    ess_threshold: float = 0.5,
) -> DiscretePosterior:
    dtype = model.initial_distribution.dtype
    n_states = model.initial_distribution.shape[0]

    def init_sample(key, _mi):
        return jr.categorical(key, jnp.log(model.initial_distribution))

    def propagate_sample(key, state, mi):
        t = jnp.array(mi - 1, dtype=dtype)
        A = model.transition_matrix(t)
        return jr.categorical(key, jnp.log(A[state]))

    def log_potential(_state_prev, state, mi):
        t = jnp.array(mi - 1, dtype=dtype)
        t_idx = jnp.array(mi - 1, dtype=jnp.int32)
        log_liks = model.emission_log_likelihood(emissions[t_idx], t)
        return log_liks[state]

    filter_obj = pf.build_filter(
        init_sample,
        propagate_sample,
        log_potential,
        n_filter_particles=n_particles,
        resampling_fn=systematic_resampling,
        ess_threshold=ess_threshold,
    )
    model_inputs = jnp.arange(emissions.shape[0] + 1)
    states = cuthbert_filter(filter_obj, model_inputs, parallel=False, key=key)

    log_w = states.log_weights[1:]
    w = jax.nn.softmax(log_w, axis=-1)
    particle_states = states.particles[1:]
    one_hot = jax.nn.one_hot(particle_states, n_states)
    filtered_probs = jnp.einsum("tn,tnk->tk", w, one_hot)

    return DiscretePosterior(
        marginal_log_likelihood=states.log_normalizing_constant[-1],
        filtered_probs=filtered_probs,
    )
