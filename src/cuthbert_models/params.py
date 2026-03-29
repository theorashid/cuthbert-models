import equinox as eqx
from jaxtyping import Array, Float


class TrainableWeights(eqx.Module):

    """Trainable weight matrix. Callable (t) -> Array."""

    matrix: Float[Array, "out in"]

    def __call__(self, _t: Float[Array, ""]) -> Float[Array, "out in"]:
        return self.matrix


class TrainableCovariance(eqx.Module):

    """
    Trainable covariance via numpyro's biject_to(constraint).

    Stores unconstrained parameters internally. The forward transform
    maps unconstrained -> PSD matrix via LowerCholeskyTransform + L @ L.T.
    """

    _unconstrained: Float[Array, " dim_tril"]
    _constraint: object = eqx.field(static=True)

    def __init__(
        self,
        matrix: Float[Array, "dim dim"],
        constraint: object | None = None,
    ):
        from numpyro.distributions import constraints  # noqa: PLC0415
        from numpyro.distributions.transforms import biject_to  # noqa: PLC0415

        if constraint is None:
            constraint = constraints.positive_definite
        self._constraint = constraint
        self._unconstrained = biject_to(constraint).inv(matrix)

    def __call__(self, _t: Float[Array, ""]) -> Float[Array, "dim dim"]:
        from numpyro.distributions.transforms import biject_to  # noqa: PLC0415

        return biject_to(self._constraint)(self._unconstrained)
