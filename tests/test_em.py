"""
EM recipe: E-step via Kalman smoother, M-step via BFGS on a closed-form ELBO.

Inspired by cuthbert's parameter_estimation_em.md example, but much simpler:
that example uses a nonlinear SSM with Gauss-Hermite quadrature for intractable
M-step integrals; here we use a linear Gaussian SSM where the ELBO has a
closed-form expression in terms of smoother sufficient statistics.

Note: the M-step ELBO is approximate -- it uses E[x_t]E[x_{t-1}]^T rather than
E[x_t x_{t-1}^T] because cuthbert's smoother does not return lag-one cross-
covariances. BFGS compensates over iterations, but the per-step ELBO is biased.
"""

import jax.numpy as jnp
import jax.random as jr
from _helpers import simulate_lgssm
from jax.scipy.optimize import minimize

from cuthbert_models import Filter, Kalman, LinearGaussianSSM, smooth


def test_em_log_likelihood_improves():
    state_dim, obs_dim = 2, 1
    F_true = jnp.array([[1.0, 1.0], [0.0, 1.0]])
    Q_true = jnp.eye(state_dim) * 0.05
    H = jnp.array([[1.0, 0.0]])
    R = jnp.eye(obs_dim) * 0.1
    m0 = jnp.array([0.0, 1.0])
    P0 = jnp.eye(state_dim)

    _, emissions = simulate_lgssm(m0, F_true, Q_true, H, R, 100, jr.key(0))

    F_hat = F_true + 0.2 * jr.normal(jr.key(1), F_true.shape)

    log_liks = []

    for _epoch in range(10):
        model = LinearGaussianSSM(
            initial_mean=m0,
            initial_covariance=P0,
            dynamics_weights=lambda _t, F=F_hat: F,
            dynamics_covariance=lambda _t: Q_true,
            emission_weights=lambda _t: H,
            emission_covariance=lambda _t: R,
        )
        with Filter(Kalman()):
            result = smooth(model, emissions)
        log_liks.append(float(result.marginal_log_likelihood))

        smoothed_means = result.smoothed_means
        smoothed_covs = result.smoothed_covariances

        def neg_elbo(
            F_flat,
            smoothed_means=smoothed_means,
            smoothed_covs=smoothed_covs,
        ):
            F = F_flat.reshape(state_dim, state_dim)
            Exp = smoothed_means[:-1]
            Exn = smoothed_means[1:]
            Vxp = smoothed_covs[:-1]

            Q_inv = jnp.linalg.solve(Q_true, jnp.eye(state_dim))
            residuals = Exn - jnp.einsum("ij,tj->ti", F, Exp)
            mahal = jnp.einsum("ti,ij,tj->", residuals, Q_inv, residuals)
            trace_term = jnp.trace(
                Q_inv @ jnp.einsum("ij,tkj->tik", F, Vxp).sum(0) @ F.T,
            )
            n = Exp.shape[0]
            _, logdet = jnp.linalg.slogdet(Q_true)
            return 0.5 * (n * logdet + mahal + trace_term)

        result_opt = minimize(neg_elbo, F_hat.ravel(), method="BFGS")
        F_hat = result_opt.x.reshape(state_dim, state_dim)

    assert log_liks[-1] > log_liks[0]
    assert jnp.allclose(F_hat, F_true, atol=0.3)
