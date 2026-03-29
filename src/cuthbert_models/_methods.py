from dataclasses import dataclass

from jax import Array


@dataclass(frozen=True)
class Kalman:
    parallel: bool = False


@dataclass(frozen=True)
class EKF:
    pass


@dataclass(frozen=True)
class UKF:
    pass


@dataclass(frozen=True)
class Forward:
    parallel: bool = False


@dataclass(frozen=True)
class Particle:
    key: Array
    n_particles: int = 100
    ess_threshold: float = 0.5
