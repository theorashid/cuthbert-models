"""
EM recipe: E-step via smoother, M-step via numerical optimisation of the ELBO.

Based on cuthbert's parameter_estimation_em.md example. Uses the smoother for
the E-step and jax.scipy.optimize.minimize for the M-step.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.optimize import minimize

from cuthbert_models import Kalman, LinearGaussianSSM


def _simulate(key, F, Q, H, R, m0, n_time):
    state_dim, obs_dim = F.shape[0], H.shape[0]

    def step(state, key):
        k1, k2 = jr.split(key)
        next_state = F @ state + jr.multivariate_normal(
            k1, jnp.zeros(state_dim), Q,
        )
        obs = H @ next_state + jr.multivariate_normal(k2, jnp.zeros(obs_dim), R)
        return next_state, obs

    _, emissions = jax.lax.scan(step, m0, jr.split(key, n_time))
    return emissions


def test_em_log_likelihood_improves():
    """EM with numerical M-step should monotonically increase log-likelihood."""
    state_dim, obs_dim = 2, 1
    F_true = jnp.array([[1.0, 1.0], [0.0, 1.0]])
    Q_true = jnp.eye(state_dim) * 0.05
    H = jnp.array([[1.0, 0.0]])
    R = jnp.eye(obs_dim) * 0.1
    m0 = jnp.array([0.0, 1.0])
    P0 = jnp.eye(state_dim)

    emissions = _simulate(jr.key(0), F_true, Q_true, H, R, m0, 100)

    # Pack F into a flat vector for optimization
    F_hat = F_true + 0.2 * jr.normal(jr.key(1), F_true.shape)

    log_liks = []

    for _epoch in range(10):
        # --- E-step: run smoother to get approximate posterior ---
        model = LinearGaussianSSM(
            initial_mean=m0,
            initial_covariance=P0,
            dynamics_weights=lambda _t, F=F_hat: F,
            dynamics_covariance=lambda _t: Q_true,
            emission_weights=lambda _t: H,
            emission_covariance=lambda _t: R,
        )
        result = model.smooth(emissions, method=Kalman())
        log_liks.append(float(result.marginal_log_likelihood))

        smoothed_means = result.smoothed_means
        smoothed_covs = result.smoothed_covariances

        # --- M-step: maximise ELBO over F numerically ---
        def neg_elbo(
            F_flat,
            smoothed_means=smoothed_means,
            smoothed_covs=smoothed_covs,
        ):
            F = F_flat.reshape(state_dim, state_dim)
            Exp = smoothed_means[:-1]
            Exn = smoothed_means[1:]
            Vxp = smoothed_covs[:-1]

            # E[||x_t - F x_{t-1}||^2_{Q^{-1}}]
            Q_inv = jnp.linalg.solve(Q_true, jnp.eye(state_dim))
            residuals = Exn - jnp.einsum("ij,tj->ti", F, Exp)
            mahal = jnp.einsum(
                "ti,ij,tj->", residuals, Q_inv, residuals,
            )
            trace_term = jnp.trace(
                Q_inv @ jnp.einsum("ij,tkj->tik", F, Vxp).sum(0) @ F.T,
            )
            n = Exp.shape[0]
            _, logdet = jnp.linalg.slogdet(Q_true)
            return 0.5 * (n * logdet + mahal + trace_term)

        result_opt = minimize(
            neg_elbo, F_hat.ravel(), method="BFGS",
        )
        F_hat = result_opt.x.reshape(state_dim, state_dim)

    # Log-likelihood should improve
    assert log_liks[-1] > log_liks[0]

    # F should approach truth
    assert jnp.allclose(F_hat, F_true, atol=0.3)
