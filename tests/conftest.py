import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float

jax.config.update("jax_enable_x64", val=True)


@pytest.fixture(scope="session")
def linear_motion_params():
    dt = 1.0
    F = jnp.array([[1.0, dt], [0.0, 1.0]])
    H = jnp.array([[1.0, 0.0]])
    Q = jnp.eye(2) * 1e-10
    R = jnp.eye(1) * 1e-10
    m0 = jnp.array([0.0, 1.0])
    P0 = jnp.eye(2) * 1e-10
    return m0, P0, F, Q, H, R


@pytest.fixture(scope="session")
def noisy_motion_params():
    dt = 1.0
    F = jnp.array([[1.0, dt], [0.0, 1.0]])
    H = jnp.array([[1.0, 0.0]])
    Q = jnp.eye(2) * 0.05
    R = jnp.eye(1) * 0.1
    m0 = jnp.array([0.0, 1.0])
    P0 = jnp.eye(2)
    return m0, P0, F, Q, H, R


def simulate_lgssm(
    m0: Float[Array, " state"],
    F: Float[Array, "state state"],
    Q: Float[Array, "state state"],
    H: Float[Array, "obs state"],
    R: Float[Array, "obs obs"],
    n_time: int,
    key: Array,
) -> tuple[Float[Array, "time state"], Float[Array, "time obs"]]:
    state_dim = F.shape[0]
    obs_dim = H.shape[0]

    def step(state, key):
        k_q, k_r = jax.random.split(key)
        q_noise = jax.random.multivariate_normal(k_q, jnp.zeros(state_dim), Q)
        next_state = F @ state + q_noise
        r_noise = jax.random.multivariate_normal(k_r, jnp.zeros(obs_dim), R)
        obs = H @ next_state + r_noise
        return next_state, (next_state, obs)

    keys = jax.random.split(key, n_time)
    _, (states, emissions) = jax.lax.scan(step, m0, keys)
    return states, emissions


@pytest.fixture(scope="session")
def linear_motion_data(linear_motion_params):
    m0, _P0, F, Q, H, R = linear_motion_params
    states, emissions = simulate_lgssm(m0, F, Q, H, R, 10, jax.random.key(0))
    return linear_motion_params, emissions, states


@pytest.fixture(scope="session")
def noisy_motion_data(noisy_motion_params):
    m0, _P0, F, Q, H, R = noisy_motion_params
    states, emissions = simulate_lgssm(m0, F, Q, H, R, 500, jax.random.key(2))
    return noisy_motion_params, emissions, states
