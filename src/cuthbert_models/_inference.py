# Bridge from model objects to cuthbert callbacks.
#
# Each function converts a model's callable fields into the callback
# protocols that cuthbert expects, then runs the filter.
#
# cuthbert's native parameterisation is Cholesky factors. We decompose
# the user's full covariance matrices here. If the model stored Cholesky
# factors directly, these decompositions could be skipped.

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
from jaxtyping import Array, Float

from cuthbert_models._types import Posterior, SmoothedPosterior
from cuthbert_models.hmm import HMM
from cuthbert_models.linear_gaussian import LinearGaussianSSM
from cuthbert_models.nonlinear_gaussian import NonlinearGaussianSSM


def _gaussian_posterior(states) -> Posterior:
    filtered_means = states.mean[1:]
    filtered_chol_covs = states.chol_cov[1:]
    filtered_covariances = (
        filtered_chol_covs @ jnp.matrix_transpose(filtered_chol_covs)
    )
    marginal_log_likelihood = states.log_normalizing_constant[-1]
    return Posterior(
        marginal_log_likelihood=marginal_log_likelihood,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covariances,
    )


# --- Linear Gaussian: exact Kalman via cuthbert.gaussian.kalman ---


def infer_kalman(
    model: LinearGaussianSSM,
    emissions: Float[Array, "time obs"],
    *,
    parallel: bool = False,
) -> Posterior:
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

    filter_obj = kalman.build_filter(
        get_init_params,
        get_dynamics_params,
        get_observation_params,
    )
    n_time = emissions.shape[0]
    model_inputs = jnp.arange(n_time + 1)
    states = cuthbert_filter(filter_obj, model_inputs, parallel=parallel)
    return _gaussian_posterior(states)


# --- Nonlinear Gaussian: EKF via cuthbert.gaussian.taylor ---


def infer_ekf(
    model: NonlinearGaussianSSM,
    emissions: Float[Array, "time obs"],
) -> Posterior:
    dtype = model.initial_mean.dtype
    state_dim = model.initial_mean.shape[0]
    m0 = model.initial_mean
    P0_inv = jnp.linalg.solve(model.initial_covariance, jnp.eye(state_dim, dtype=dtype))

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

    filter_obj = taylor_filter.build_filter(
        get_init_log_density,
        get_dynamics_log_density,
        get_observation_func,
        associative=False,
    )
    n_time = emissions.shape[0]
    model_inputs = jnp.arange(n_time + 1)
    states = cuthbert_filter(filter_obj, model_inputs, parallel=False)
    return _gaussian_posterior(states)


# --- Nonlinear Gaussian: UKF via cuthbert.gaussian.moments ---


def infer_ukf(
    model: NonlinearGaussianSSM,
    emissions: Float[Array, "time obs"],
) -> Posterior:
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

    filter_obj = moments_filter.build_filter(
        get_init_params,
        get_dynamics_moments,
        get_observation_moments,
        associative=False,
    )
    n_time = emissions.shape[0]
    model_inputs = jnp.arange(n_time + 1)
    states = cuthbert_filter(filter_obj, model_inputs, parallel=False)
    return _gaussian_posterior(states)


# --- HMM: forward algorithm via cuthbert.discrete ---


def infer_forward(
    model: HMM,
    emissions: Float[Array, "time ..."],
    *,
    parallel: bool = False,
) -> Posterior:
    dtype = model.initial_distribution.dtype
    n_time = emissions.shape[0]

    def get_init_dist(_mi):
        return model.initial_distribution

    def get_transition_matrix(mi):
        t = jnp.array(mi - 1, dtype=dtype)
        return model.transition_matrix(t)

    def get_obs_log_likelihoods(mi):
        t = jnp.array(mi - 1, dtype=dtype)
        t_idx = jnp.array(mi - 1, dtype=jnp.int32)
        return model.emission_log_likelihood(emissions[t_idx], t)

    filter_obj = discrete_filter.build_filter(
        get_init_dist,
        get_transition_matrix,
        get_obs_log_likelihoods,
    )
    model_inputs = jnp.arange(n_time + 1)
    states = cuthbert_filter(filter_obj, model_inputs, parallel=parallel)

    filtered_probs = states.dist[1:]
    marginal_log_likelihood = states.log_normalizing_constant[-1]

    return Posterior(
        marginal_log_likelihood=marginal_log_likelihood,
        filtered_means=filtered_probs,
        filtered_covariances=jnp.zeros_like(filtered_probs),
    )


# --- Particle filter: bootstrap PF via cuthbert.smc ---


def infer_particle_gaussian(
    model: NonlinearGaussianSSM,
    emissions: Float[Array, "time obs"],
    *,
    key: jax.Array,
    n_particles: int = 100,
    ess_threshold: float = 0.5,
) -> Posterior:
    """Bootstrap particle filter for NonlinearGaussianSSM."""
    dtype = model.initial_mean.dtype
    state_dim = model.initial_mean.shape[0]
    m0 = model.initial_mean
    P0 = model.initial_covariance
    chol_P0 = jnp.linalg.cholesky(P0)

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
    n_time = emissions.shape[0]
    model_inputs = jnp.arange(n_time + 1)
    states = cuthbert_filter(filter_obj, model_inputs, parallel=False, key=key)

    # Weighted mean of particles
    log_w = states.log_weights[1:]
    w = jax.nn.softmax(log_w, axis=-1)
    particles = states.particles[1:]
    filtered_means = jnp.einsum("tn,tn...->t...", w, particles)
    marginal_log_likelihood = states.log_normalizing_constant[-1]

    return Posterior(
        marginal_log_likelihood=marginal_log_likelihood,
        filtered_means=filtered_means,
        filtered_covariances=jnp.zeros_like(filtered_means),
    )


def infer_particle_hmm(
    model: HMM,
    emissions: Float[Array, "time ..."],
    *,
    key: jax.Array,
    n_particles: int = 100,
    ess_threshold: float = 0.5,
) -> Posterior:
    """Bootstrap particle filter for HMMs."""
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
    n_time = emissions.shape[0]
    model_inputs = jnp.arange(n_time + 1)
    states = cuthbert_filter(filter_obj, model_inputs, parallel=False, key=key)

    # Convert particle states to probability estimates
    log_w = states.log_weights[1:]
    w = jax.nn.softmax(log_w, axis=-1)
    particle_states = states.particles[1:]
    one_hot = jax.nn.one_hot(particle_states, n_states)
    filtered_probs = jnp.einsum("tn,tnk->tk", w, one_hot)
    marginal_log_likelihood = states.log_normalizing_constant[-1]

    return Posterior(
        marginal_log_likelihood=marginal_log_likelihood,
        filtered_means=filtered_probs,
        filtered_covariances=jnp.zeros_like(filtered_probs),
    )


# --- Smoothers ---


def _gaussian_smoothed_posterior(
    filter_states, smoother_states,
) -> SmoothedPosterior:
    filtered = _gaussian_posterior(filter_states)
    smoothed_means = smoother_states.mean[1:]
    smoothed_chol_covs = smoother_states.chol_cov[1:]
    smoothed_covariances = (
        smoothed_chol_covs @ jnp.matrix_transpose(smoothed_chol_covs)
    )
    return SmoothedPosterior(
        marginal_log_likelihood=filtered.marginal_log_likelihood,
        filtered_means=filtered.filtered_means,
        filtered_covariances=filtered.filtered_covariances,
        smoothed_means=smoothed_means,
        smoothed_covariances=smoothed_covariances,
    )


def smooth_kalman(
    model: LinearGaussianSSM,
    emissions: Float[Array, "time obs"],
    *,
    parallel: bool = False,
) -> SmoothedPosterior:
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

    filter_obj = kalman.build_filter(
        get_init_params, get_dynamics_params, get_observation_params,
    )
    smoother_obj = kalman.build_smoother(get_dynamics_params)

    n_time = emissions.shape[0]
    model_inputs = jnp.arange(n_time + 1)
    filter_states = cuthbert_filter(filter_obj, model_inputs, parallel=parallel)
    smoother_states = cuthbert_smoother(
        smoother_obj, filter_states, parallel=parallel,
    )
    return _gaussian_smoothed_posterior(filter_states, smoother_states)


def smooth_ekf(
    model: NonlinearGaussianSSM,
    emissions: Float[Array, "time obs"],
) -> SmoothedPosterior:
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

    filter_obj = taylor_filter.build_filter(
        get_init_log_density, get_dynamics_log_density, get_observation_func,
        associative=False,
    )
    smoother_obj = taylor_smoother_mod.build_smoother(get_dynamics_log_density)

    n_time = emissions.shape[0]
    model_inputs = jnp.arange(n_time + 1)
    filter_states = cuthbert_filter(filter_obj, model_inputs, parallel=False)
    smoother_states = cuthbert_smoother(
        smoother_obj, filter_states, parallel=False,
    )
    return _gaussian_smoothed_posterior(filter_states, smoother_states)


def smooth_ukf(
    model: NonlinearGaussianSSM,
    emissions: Float[Array, "time obs"],
) -> SmoothedPosterior:
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

    filter_obj = moments_filter.build_filter(
        get_init_params, get_dynamics_moments, get_observation_moments,
        associative=False,
    )
    smoother_obj = moments_smoother_mod.build_smoother(get_dynamics_moments)

    n_time = emissions.shape[0]
    model_inputs = jnp.arange(n_time + 1)
    filter_states = cuthbert_filter(filter_obj, model_inputs, parallel=False)
    smoother_states = cuthbert_smoother(
        smoother_obj, filter_states, parallel=False,
    )
    return _gaussian_smoothed_posterior(filter_states, smoother_states)


def smooth_forward(
    model: HMM,
    emissions: Float[Array, "time ..."],
    *,
    parallel: bool = False,
) -> SmoothedPosterior:
    dtype = model.initial_distribution.dtype
    n_time = emissions.shape[0]

    def get_init_dist(_mi):
        return model.initial_distribution

    def get_transition_matrix(mi):
        t = jnp.array(mi - 1, dtype=dtype)
        return model.transition_matrix(t)

    def get_obs_log_likelihoods(mi):
        t = jnp.array(mi - 1, dtype=dtype)
        t_idx = jnp.array(mi - 1, dtype=jnp.int32)
        return model.emission_log_likelihood(emissions[t_idx], t)

    filter_obj = discrete_filter.build_filter(
        get_init_dist, get_transition_matrix, get_obs_log_likelihoods,
    )
    smoother_obj = discrete_smoother_mod.build_smoother(get_transition_matrix)

    model_inputs = jnp.arange(n_time + 1)
    filter_states = cuthbert_filter(filter_obj, model_inputs, parallel=parallel)
    smoother_states = cuthbert_smoother(
        smoother_obj, filter_states, parallel=parallel,
    )

    filtered_probs = filter_states.dist[1:]
    smoothed_probs = smoother_states.dist[1:]
    marginal_log_likelihood = filter_states.log_normalizing_constant[-1]

    return SmoothedPosterior(
        marginal_log_likelihood=marginal_log_likelihood,
        filtered_means=filtered_probs,
        filtered_covariances=jnp.zeros_like(filtered_probs),
        smoothed_means=smoothed_probs,
        smoothed_covariances=jnp.zeros_like(smoothed_probs),
    )
