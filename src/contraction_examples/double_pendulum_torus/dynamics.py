"""Planar double pendulum in absolute angles.

Both angles are measured from the downward vertical (counter-clockwise
positive): theta1 for the inner link, theta2 for the outer link. The
configuration (theta1, theta2) lives on the torus T^2 = S^1 x S^1.

Euler-Lagrange equations, with Delta = theta1 - theta2 and
h = m2 l1 l2 sin(Delta):

    (m1+m2) l1^2 th1'' + m2 l1 l2 cos(Delta) th2'' = -h th2'^2 - (m1+m2) g l1 sin(th1)
    m2 l1 l2 cos(Delta) th1'' + m2 l2^2 th2''       = +h th1'^2 - m2 g l2 sin(th2)

(optionally with viscous joint damping added to the right-hand sides).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class DoublePendulumParams:
    gravity: float = 9.81  # [m/s^2]
    m1: float = 1.0        # inner bob mass [kg]
    m2: float = 1.0        # outer bob mass [kg]
    l1: float = 0.5        # inner link length [m]
    l2: float = 0.5        # outer link length [m]
    damping1: float = 0.0  # viscous damping at the shoulder [1/s]
    damping2: float = 0.0  # viscous damping at the elbow [1/s]


@dataclass(frozen=True)
class Trajectory:
    t: np.ndarray
    theta1: np.ndarray
    theta2: np.ndarray
    theta1_dot: np.ndarray
    theta2_dot: np.ndarray


def vector_field(t: float, state: np.ndarray, p: DoublePendulumParams) -> list[float]:
    th1, th2, w1, w2 = state
    delta = th1 - th2
    a = (p.m1 + p.m2) * p.l1**2
    b = p.m2 * p.l1 * p.l2 * np.cos(delta)
    c = p.m2 * p.l2**2
    h = p.m2 * p.l1 * p.l2 * np.sin(delta)
    rhs1 = -h * w2**2 - (p.m1 + p.m2) * p.gravity * p.l1 * np.sin(th1) \
        - p.damping1 * w1
    rhs2 = h * w1**2 - p.m2 * p.gravity * p.l2 * np.sin(th2) - p.damping2 * w2
    det = a * c - b * b
    return [w1, w2, (c * rhs1 - b * rhs2) / det, (a * rhs2 - b * rhs1) / det]


def energy(state: np.ndarray, p: DoublePendulumParams) -> float:
    """Total mechanical energy (kinetic + potential); conserved when undamped."""
    th1, th2, w1, w2 = state
    kinetic = (
        0.5 * (p.m1 + p.m2) * p.l1**2 * w1**2
        + 0.5 * p.m2 * p.l2**2 * w2**2
        + p.m2 * p.l1 * p.l2 * w1 * w2 * np.cos(th1 - th2)
    )
    potential = (
        -(p.m1 + p.m2) * p.gravity * p.l1 * np.cos(th1)
        - p.m2 * p.gravity * p.l2 * np.cos(th2)
    )
    return kinetic + potential


def simulate(
    params: DoublePendulumParams,
    theta1_0: float,
    theta2_0: float,
    theta1_dot0: float,
    theta2_dot0: float,
    t_end: float,
    fps: int,
) -> Trajectory:
    """Integrate the double pendulum and sample it at the animation frame rate."""
    t_eval = np.arange(0.0, t_end, 1.0 / fps)
    sol = solve_ivp(
        vector_field,
        (0.0, t_end),
        [theta1_0, theta2_0, theta1_dot0, theta2_dot0],
        args=(params,),
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-11,
        atol=1e-11,
    )
    return Trajectory(t=sol.t, theta1=sol.y[0], theta2=sol.y[1],
                      theta1_dot=sol.y[2], theta2_dot=sol.y[3])
