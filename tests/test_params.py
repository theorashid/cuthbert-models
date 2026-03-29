import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu

from cuthbert_models import TrainableCovariance, TrainableWeights


def test_trainable_weights_roundtrip():
    W = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    tw = TrainableWeights(W)
    assert jnp.array_equal(tw(jnp.array(0.0)), W)


def test_trainable_covariance_properties():
    dim = 3
    cov = TrainableCovariance(jnp.eye(dim))

    cov_matrix = cov(jnp.array(0.0))
    assert cov_matrix.shape == (dim, dim)
    assert jnp.allclose(cov_matrix, cov_matrix.T, atol=1e-6)
    assert jnp.all(jnp.linalg.eigvalsh(cov_matrix) > 0)
    assert jnp.allclose(cov_matrix, jnp.eye(dim), atol=1e-4)


def test_trainable_covariance_single_leaf():
    cov = TrainableCovariance(jnp.eye(3))
    dynamic_leaves = jtu.tree_leaves(eqx.partition(cov, eqx.is_array)[0])
    assert len(dynamic_leaves) == 1
