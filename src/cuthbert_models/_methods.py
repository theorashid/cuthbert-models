from collections.abc import Callable
from dataclasses import dataclass

from jax import Array
from jaxtyping import Key


@dataclass(frozen=True)
class Kalman:
    parallel: bool = False


@dataclass(frozen=True)
class EKF:
    linearization: str


@dataclass(frozen=True)
class Forward:
    parallel: bool = False


@dataclass(frozen=True)
class Particle:
    key: Key[Array, ""]
    resampling_fn: Callable
    n_particles: int = 100
