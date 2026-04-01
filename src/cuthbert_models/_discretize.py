from dataclasses import dataclass


@dataclass(frozen=True)
class VanLoan:
    pass


# @dataclass(frozen=True)
# class AdaptiveODE:
#     """Adaptive ODE solver discretisation via diffrax.
#
#     Numerically integrates the moment ODEs (dA/dt, dQ/dt) between
#     observation times. Works for both linear and nonlinear models,
#     but introduces numerical integration error controlled by tolerances.
#
#     Requires diffrax as an optional dependency.
#     """
#
#     dt0: float = 0.01
#     max_steps: int = 1000


@dataclass(frozen=True)
class EulerMaruyama:
    pass
