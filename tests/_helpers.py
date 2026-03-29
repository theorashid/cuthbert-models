import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Key


def simulate_lgssm(
    m0: Float[Array, " s"],
    F: Float[Array, "s s"],
    Q: Float[Array, "s s"],
    H: Float[Array, "o s"],
    R: Float[Array, "o o"],
    n_time: int,
    key: Key[Array, ""],
) -> tuple[Float[Array, "t s"], Float[Array, "t o"]]:
    state_dim, obs_dim = F.shape[0], H.shape[0]

    def step(state, key):
        k1, k2 = jr.split(key)
        next_state = F @ state + jr.multivariate_normal(k1, jnp.zeros(state_dim), Q)
        obs = H @ next_state + jr.multivariate_normal(k2, jnp.zeros(obs_dim), R)
        return next_state, (next_state, obs)

    _, (states, emissions) = jax.lax.scan(step, m0, jr.split(key, n_time))
    return states, emissions


def random_lgssm_args(
    key: Key[Array, ""], state_dim: int = 2, obs_dim: int = 1, n_time: int = 50,
) -> tuple:
    keys = jr.split(key, 7)
    m0 = jr.normal(keys[0], (state_dim,))
    P0 = jnp.eye(state_dim) * jr.uniform(keys[1], minval=0.5, maxval=1.5)
    F = jnp.eye(state_dim) + 0.1 * jr.normal(keys[2], (state_dim, state_dim))
    Q = jnp.eye(state_dim) * jr.uniform(keys[3], minval=0.01, maxval=0.1)
    H = jr.normal(keys[4], (obs_dim, state_dim))
    R = jnp.eye(obs_dim) * jr.uniform(keys[5], minval=0.05, maxval=0.2)
    _, emissions = simulate_lgssm(m0, F, Q, H, R, n_time, keys[6])
    return (m0, P0, F, Q, H, R), emissions
