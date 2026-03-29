import equinox as eqx
from jax import Array


class Kalman(eqx.Module):
    parallel: bool = False


class EKF(eqx.Module):
    pass


class UKF(eqx.Module):
    pass


class Forward(eqx.Module):
    parallel: bool = False


class Particle(eqx.Module):
    key: Array
    n_particles: int = 100
    ess_threshold: float = 0.5
