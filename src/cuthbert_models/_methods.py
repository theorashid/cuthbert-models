from dataclasses import dataclass

from jaxtyping import PRNGKeyArray


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
    key: PRNGKeyArray
    n_particles: int = 100
    ess_threshold: float = 0.5
