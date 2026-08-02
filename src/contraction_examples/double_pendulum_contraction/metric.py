"""The mechanical contraction metric for the damped double pendulum.

Dynamics: M(q) q_dd + C(q, q_dot) q_dot + g(q) + D q_dot = 0 on T^2.

Is the metric "the mass matrix"? Almost. The kinetic-energy metric M(q)
is the natural Riemannian metric on the configuration torus, and it is
the VELOCITY block of the contraction metric — but on its own (the energy
metric diag(K0, M0), i.e. eps = 0) it only gives semi-contraction:
d/dt of the energy of a virtual displacement is -2 dq_dot^T D dq_dot,
which vanishes whenever dq_dot = 0. The classic cross-term repair

    P = [[K0 + eps D, eps M0],
        [eps M0,      M0    ]]

(with M0, K0 the mass and stiffness matrices at the hanging equilibrium)
satisfies, for the LINEARIZATION M0 q_dd + D q_dot + K0 q = 0, the exact
identity

    P A + A^T P = -2 blkdiag(eps K0, D - eps M0),

which is negative definite whenever K0 > 0 and D - eps M0 > 0: strict
contraction. For the nonlinear pendulum the same constant P certifies
contraction on the region where P J(x) + J(x)^T P < 0, J the state
Jacobian — checked numerically here, both on the q_dot = 0 slice of the
torus (for the painted region) and along the segments between simulated
trajectory pairs (the actual certificate for pairwise-distance decay).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from ..double_pendulum_torus.dynamics import DoublePendulumParams, vector_field


def mass_matrix_at_origin(p: DoublePendulumParams) -> np.ndarray:
    return np.array([
        [(p.m1 + p.m2) * p.l1**2, p.m2 * p.l1 * p.l2],
        [p.m2 * p.l1 * p.l2, p.m2 * p.l2**2],
    ])


def stiffness_at_origin(p: DoublePendulumParams) -> np.ndarray:
    return np.diag([(p.m1 + p.m2) * p.gravity * p.l1, p.m2 * p.gravity * p.l2])


def damping_matrix(p: DoublePendulumParams) -> np.ndarray:
    return np.diag([p.damping1, p.damping2])


def structured_metric(p: DoublePendulumParams, eps: float) -> np.ndarray:
    """P = [[K0 + eps D, eps M0], [eps M0, M0]]; raises if not admissible."""
    M0, K0, D = mass_matrix_at_origin(p), stiffness_at_origin(p), damping_matrix(p)
    P = np.block([[K0 + eps * D, eps * M0], [eps * M0, M0]])
    if not np.all(np.linalg.eigvalsh(P) > 0):
        raise ValueError(f"P not positive definite for eps={eps}")
    if not np.all(np.linalg.eigvalsh(D - eps * M0) > 0):
        raise ValueError(f"D - eps*M0 not positive definite for eps={eps}; "
                         "reduce --eps")
    return P


def jacobian(p: DoublePendulumParams, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """State Jacobian of the nonlinear flow by central differences."""
    J = np.zeros((4, 4))
    for k in range(4):
        e = np.zeros(4)
        e[k] = h
        J[:, k] = (np.array(vector_field(0.0, x + e, p))
                   - np.array(vector_field(0.0, x - e, p))) / (2.0 * h)
    return J


def slice_field(p: DoublePendulumParams, P: np.ndarray,
                n: int = 81) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """lambda_max(PJ + J^T P) over the q_dot = 0 slice of the torus."""
    th = np.linspace(-np.pi, np.pi, n)
    T1, T2 = np.meshgrid(th, th)
    LAM = np.zeros_like(T1)
    for i in range(n):
        for j in range(n):
            J = jacobian(p, np.array([T1[i, j], T2[i, j], 0.0, 0.0]))
            LAM[i, j] = np.linalg.eigvalsh(P @ J + J.T @ P).max()
    return T1, T2, LAM


def segment_certificate(p: DoublePendulumParams, P: np.ndarray,
                        Z: np.ndarray, t_stride: int = 8,
                        n_interp: int = 5) -> tuple[float, float]:
    """Check P J + J^T P < 0 along segments between all trajectory pairs.

    Z has shape (N, n_t, 4). Returns (worst lambda_max, worst guaranteed
    distance rate); the rate is -inf if the certificate fails anywhere.
    """
    N = Z.shape[0]
    worst, rate_min = -np.inf, np.inf
    svals = np.linspace(0.0, 1.0, n_interp)
    for ti in range(0, Z.shape[1], t_stride):
        states = Z[:, ti]
        for i in range(N):
            for j in range(i + 1, N):
                for s in svals:
                    x = (1.0 - s) * states[i] + s * states[j]
                    J = jacobian(p, x)
                    S = -(P @ J + J.T @ P)
                    ev_min = np.linalg.eigvalsh(S).min()
                    worst = max(worst, -ev_min)
                    if ev_min > 0.0 and rate_min != -np.inf:
                        rate_min = min(rate_min,
                                       eigh(S, P, eigvals_only=True)[0])
                    else:
                        rate_min = -np.inf
    return worst, 0.5 * rate_min
