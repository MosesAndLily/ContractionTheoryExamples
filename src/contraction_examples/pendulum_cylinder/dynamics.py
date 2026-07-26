"""Damped simple pendulum dynamics.

The configuration is the angle theta on the circle S^1 (measured from the
downward vertical, counter-clockwise positive) and the state (theta, theta_dot)
lives on the tangent bundle TS^1 = S^1 x R:

    theta_ddot = -(g / L) sin(theta) - c * theta_dot
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class PendulumParams:
    gravity: float = 9.81  # [m/s^2]
    length: float = 1.0    # [m]
    damping: float = 0.55  # viscous damping coefficient c [1/s]


@dataclass(frozen=True)
class Trajectory:
    t: np.ndarray
    theta: np.ndarray
    theta_dot: np.ndarray


def vector_field(t: float, state: np.ndarray, params: PendulumParams) -> list[float]:
    theta, theta_dot = state
    theta_ddot = -(params.gravity / params.length) * np.sin(theta) - params.damping * theta_dot
    return [theta_dot, theta_ddot]


def simulate(
    params: PendulumParams,
    theta0: float,
    theta_dot0: float,
    t_end: float,
    fps: int,
) -> Trajectory:
    """Integrate the pendulum and sample it at the animation frame rate."""
    t_eval = np.arange(0.0, t_end, 1.0 / fps)
    sol = solve_ivp(
        vector_field,
        (0.0, t_end),
        [theta0, theta_dot0],
        args=(params,),
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-10,
        atol=1e-10,
    )
    return Trajectory(t=sol.t, theta=sol.y[0], theta_dot=sol.y[1])
