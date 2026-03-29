# Bridge from model objects to cuthbert callbacks.
#
# Each function converts a model's callable fields into the callback
# protocols that cuthbert expects, then runs the filter.
#
# cuthbert's native parameterisation is Cholesky factors. We decompose
# the user's full covariance matrices here. If the model stored Cholesky
# factors directly, these decompositions could be skipped.

import jax.numpy as jnp
from cuthbert import filter as cuthbert_filter
from cuthbert.discrete import filter as discrete_filter
from cuthbert.gaussian import kalman
from cuthbert.gaussian.moments import filter as moments_filter
from cuthbert.gaussian.taylor import filter as taylor_filter
from jaxtyping import Array, Float

from cuthbert_models._types import Posterior
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
        obs_dim = emissions.shape[-1]
        R_t = model.emission_covariance(t)
        R_t_inv = jnp.linalg.solve(R_t, jnp.eye(obs_dim, dtype=dtype))
        y = emissions[t_idx]

        def obs_log_density(x, y_obs):
            predicted = model.emission_fn(x, t)
            diff = y_obs - predicted
            return -0.5 * diff @ R_t_inv @ diff

        x_lin = state.mean
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
        R_t = model.emission_covariance(t)
        chol_R_t = jnp.linalg.cholesky(R_t)

        def mean_and_chol_cov(x):
            return model.emission_fn(x, t), chol_R_t

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
