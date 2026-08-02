"""The two metrics of the story.

Identity metric: V_I(z) = ||z||^2. Along the flow,
d/dt V_I = z^T (A + A^T) z, and whenever A + A^T has a non-negative
eigenvalue the Euclidean norm fails to contract — it stalls, or (when the
symmetric part is indefinite, as for the default k = 2) transiently grows.

Lyapunov metric: P solves the Lyapunov equation

    P A + A^T P = -I,   P = P^T > 0,

so V_P(z) = z^T P z satisfies d/dt V_P = -||z||^2 < 0 away from the
origin: the same flow contracts in the P-weighted norm at every instant,
with rate at least 1 / lambda_max(P). For A = [[0, 1], [-2, -1]] the
solution is P = [[7/4, 1/4], [1/4, 3/4]].
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_lyapunov


def lyapunov_P(A: np.ndarray) -> np.ndarray:
    """Solve P A + A^T P = -I for the (symmetrized) positive-definite P."""
    P = solve_continuous_lyapunov(A.T, -np.eye(A.shape[0]))
    P = 0.5 * (P + P.T)
    assert np.allclose(P @ A + A.T @ P, -np.eye(A.shape[0])), \
        "Lyapunov solve failed — is A Hurwitz?"
    assert np.all(np.linalg.eigvalsh(P) > 0), "P is not positive definite"
    return P


def quad_form(z: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Row-wise quadratic form z^T M z for an (n, 2) array of states."""
    return np.einsum("ij,jk,ik->i", z, M, z)


def level_set_points(M: np.ndarray, c: float, n: int = 200) -> np.ndarray:
    """The ellipse { s : s^T M s = c } as a (2, n) point array.

    With M^-1 = L L^T (Cholesky), s = sqrt(c) L u maps the unit circle
    onto the level set: s^T M s = c u^T L^T M L u = c.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    L = np.linalg.cholesky(np.linalg.inv(M))
    return np.sqrt(c) * (L @ circle)
