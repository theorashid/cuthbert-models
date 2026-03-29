import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from cuthbert_models import LinearGaussianSSM, TrainableWeights


def test_mle_learns_dynamics():
    """MLE with optax recovers the dynamics matrix from noisy data."""
    state_dim, obs_dim = 2, 1
    dt = 1.0
    F_true = jnp.array([[1.0, dt], [0.0, 1.0]])
    Q_true = jnp.eye(state_dim) * 0.05
    H = jnp.array([[1.0, 0.0]])
    R = jnp.eye(obs_dim) * 0.1
    m0 = jnp.array([0.0, 1.0])
    P0 = jnp.eye(state_dim)

    key = jax.random.key(0)

    def simulate(key, n_time):
        def step(state, key):
            k_q, k_r = jax.random.split(key)
            q = jax.random.multivariate_normal(k_q, jnp.zeros(state_dim), Q_true)
            next_state = F_true @ state + q
            r = jax.random.multivariate_normal(k_r, jnp.zeros(obs_dim), R)
            obs = H @ next_state + r
            return next_state, obs

        keys = jax.random.split(key, n_time)
        _, emissions = jax.lax.scan(step, m0, keys)
        return emissions

    emissions = simulate(key, 200)

    # Initialise with slightly perturbed dynamics (close to true).
    # Perturb init away from identity to avoid cuthbert's NaN gradient at F=I.
    F_init = F_true + jax.random.normal(jax.random.key(1), F_true.shape) * 0.05
    model = LinearGaussianSSM(
        initial_mean=m0,
        initial_covariance=P0,
        dynamics_weights=TrainableWeights(F_init),
        dynamics_covariance=lambda _t: Q_true,
        emission_weights=lambda _t: H,
        emission_covariance=lambda _t: R,
    )

    # Only train the dynamics weights, freeze everything else
    filter_spec = jax.tree.map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        lambda m: m.dynamics_weights, filter_spec, replace=True
    )
    trainable, static = eqx.partition(model, filter_spec)

    @eqx.filter_jit
    def loss_fn(trainable):
        model = eqx.combine(trainable, static)
        return -model.infer(emissions).marginal_log_likelihood

    optimiser = optax.adam(1e-2)
    opt_state = optimiser.init(eqx.filter(trainable, eqx.is_array))

    @eqx.filter_jit
    def step(trainable, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(trainable)
        updates, opt_state = optimiser.update(grads, opt_state, trainable)
        trainable = eqx.apply_updates(trainable, updates)
        return trainable, opt_state, loss

    for _ in range(200):
        trainable, opt_state, _loss = step(trainable, opt_state)

    final_model = eqx.combine(trainable, static)
    F_learned = final_model.dynamics_weights(jnp.array(0.0))
    assert jnp.allclose(F_learned, F_true, atol=0.3)
