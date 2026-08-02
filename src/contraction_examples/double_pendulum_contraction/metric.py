"""The mechanical contraction metric for the damped double pendulum.

Dynamics: M(q) q_dd + C(q, q_dot) q_dot + g(q) + D q_dot = 0 on T^2.

Is the metric "the mass matrix"? Almost. The kinetic-energy metric M(q)
is the natural Riemannian metric on the configuration torus, and it is
the VELOCITY block of the contraction metric — but on its own (the
energy metric, eps = 0) it only gives semi-contraction: dissipation acts
only through the velocity part of a virtual displacement. The cross-term
block metric repairs this, and comes in two flavours here:

    P(q) = [[K0 + eps D, eps M(q)],
           [eps M(q),    M(q)   ]]      (state-dependent, the default)

with M(q) the actual mass matrix, or the constant variant that freezes
M(q) at the hanging equilibrium, P0 = P(0). For the LINEARIZATION the
constant variant satisfies the exact identity

    P0 A + A^T P0 = -2 blkdiag(eps K0, D - eps M0) < 0.

For the nonlinear flow with a state-dependent metric the contraction
condition acquires the metric-rate term:

    Pdot(x) + P(x) J(x) + J(x)^T P(x) < 0,   Pdot = sum_k dP/dq_k qdot_k,

which this module checks numerically — on the qdot = 0 slice of the
torus (where Pdot vanishes identically, since P depends only on q) for
the painted region, and along the straight segments between simulated
trajectory pairs, which is what the pairwise-distance decay integrates
over. Distances are correspondingly measured as segment integrals of
sqrt(dx^T P(gamma(s)) dx).
"""

from __future__ import annotations

from dataclasses import dataclass

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
    """The constant variant P0 = P(0); raises if not admissible."""
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


@dataclass(frozen=True)
class MechanicalMetric:
    """P(q) = [[K0 + eps D, eps M(q)], [eps M(q), M(q)]] on TT^2.

    ``constant=True`` freezes the mass matrix at the hanging equilibrium,
    recovering the constant-chart metric P0.
    """

    p: DoublePendulumParams
    eps: float
    constant: bool = False

    def __post_init__(self):
        # admissibility at the extremes of the mass-matrix off-diagonal
        for delta in (0.0, np.pi):
            if not np.all(np.linalg.eigvalsh(self._block(delta)) > 0):
                raise ValueError(f"P(q) not positive definite for "
                                 f"eps={self.eps}; reduce --eps")
        structured_metric(self.p, self.eps)  # also enforce D - eps*M0 > 0

    def _block(self, delta: float) -> np.ndarray:
        p = self.p
        b = p.m2 * p.l1 * p.l2 * np.cos(delta)
        Mq = np.array([[(p.m1 + p.m2) * p.l1**2, b], [b, p.m2 * p.l2**2]])
        K0, D = stiffness_at_origin(p), damping_matrix(p)
        return np.block([[K0 + self.eps * D, self.eps * Mq],
                         [self.eps * Mq, Mq]])

    def at(self, x: np.ndarray) -> np.ndarray:
        """The metric at state x (depends only on q, via theta1 - theta2)."""
        delta = 0.0 if self.constant else float(x[0] - x[1])
        return self._block(delta)

    def rate_matrix(self, x: np.ndarray,
                    h: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        """(Pdot + PJ + J^T P, P) at state x; Pdot = 0 for the constant
        variant and on the qdot = 0 slice."""
        Pq = self.at(x)
        J = jacobian(self.p, x)
        S = Pq @ J + J.T @ Pq
        if not self.constant:
            f = np.array(vector_field(0.0, x, self.p))
            S = S + (self.at(x + h * f) - self.at(x - h * f)) / (2.0 * h)
        return S, Pq


def slice_field(metric: MechanicalMetric,
                n: int = 81) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """lambda_max of the contraction condition on the qdot = 0 slice of the
    torus (the metric-rate term vanishes there)."""
    th = np.linspace(-np.pi, np.pi, n)
    T1, T2 = np.meshgrid(th, th)
    LAM = np.zeros_like(T1)
    for i in range(n):
        for j in range(n):
            S, _ = metric.rate_matrix(np.array([T1[i, j], T2[i, j], 0.0, 0.0]))
            LAM[i, j] = np.linalg.eigvalsh(S).max()
    return T1, T2, LAM


def segment_certificate(metric: MechanicalMetric, Z: np.ndarray,
                        t_stride: int = 8,
                        n_interp: int = 5) -> tuple[float, float]:
    """Check Pdot + PJ + J^T P < 0 along segments between trajectory pairs.

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
                    S, Pq = metric.rate_matrix(x)
                    ev_min = np.linalg.eigvalsh(-S).min()
                    worst = max(worst, -ev_min)
                    if ev_min > 0.0 and rate_min != -np.inf:
                        rate_min = min(rate_min,
                                       eigh(-S, Pq, eigvals_only=True)[0])
                    else:
                        rate_min = -np.inf
    return worst, 0.5 * rate_min
