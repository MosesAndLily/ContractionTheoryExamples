"""Unit-mass mass-spring-damper (m = 1) as a linear system on the phase
plane z = (x, x_dot):

    z_dot = A z,   A = [[0, 1], [-k, -c]]

Default k = 2, c = 1: eig(A) = -1/2 +- (sqrt(7)/2) i, exponentially stable
with rate 1/2. But A + A^T = [[0, -(k-1)], [-(k-1), -2c]] is INDEFINITE
(eigenvalues -1 +- sqrt(2) for the default), so the Euclidean norm of a
perturbation not only stalls at turning points — it transiently grows on
part of every oscillation. The system is not contracting in the identity
metric even though it is exponentially stable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


def make_A(k: float = 2.0, c: float = 1.0) -> np.ndarray:
    """System matrix of x_ddot = -k x - c x_dot (unit mass)."""
    return np.array([[0.0, 1.0], [-k, -c]])


@dataclass(frozen=True)
class Trajectory:
    t: np.ndarray
    z: np.ndarray  # shape (n, 2): columns are x and x_dot


def simulate(A: np.ndarray, z0: np.ndarray, t_end: float, fps: int) -> Trajectory:
    """Integrate z_dot = A z, sampled at the animation frame rate."""
    t_eval = np.arange(0.0, t_end, 1.0 / fps)
    sol = solve_ivp(
        lambda t, z: A @ z,
        (0.0, t_end),
        np.asarray(z0, dtype=float),
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
    )
    return Trajectory(t=sol.t, z=sol.y.T)


def turning_indices(traj: Trajectory) -> np.ndarray:
    """Sample indices where x_dot changes sign (turning points of x)."""
    xd = traj.z[:, 1]
    return np.where(np.diff(np.sign(xd)) != 0)[0]
