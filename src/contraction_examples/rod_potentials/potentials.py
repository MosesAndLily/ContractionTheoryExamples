"""The three potential energies of a planar two-rod arm and their sum.

Configuration q = (q1, q2) on the torus T^2 (relative joint angles, q = 0
is the arm stretched along +x). With tip position x(q) and tip orientation
phi(q) = q1 + q2, the three terms are

    U_q   = k_q  sum_i (1 - cos(q_i - q*_i))       joint space
    U_x   = 1/2 k_x ||x(q) - x*||^2                 task position
    U_phi = k_phi (1 - cos(phi(q) - phi*))          task orientation

(the cosines are the smooth periodic versions of 1/2 k (.)^2, so every term
is a genuine function on T^2). The first-order flow q_dot = -grad U / b has
the symmetric Jacobian -Hess U / b, so in the flat metric of T^2 it
contracts at rate lambda_min(Hess U) / b wherever that is positive, and by
Weyl's inequality lambda_min(sum H_i) >= sum lambda_min(H_i).

Everything here is closed form and vectorized over leading axes; the
simulation evaluates the same gradients through MuJoCo (``model.py``) and
cross-checks them against these.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TERMS = ("joint", "position", "orientation")


@dataclass(frozen=True)
class RodPotentials:
    l1: float = 1.0
    l2: float = 1.0
    k_q: float = 1.0      # joint stiffness
    k_x: float = 2.0      # task-position stiffness
    k_phi: float = 1.0    # task-orientation stiffness
    b: float = 1.0        # first-order damping: b q_dot = -grad U
    q_star: tuple[float, float] = (0.4, 1.2)

    # ---------------------------------------------------------- kinematics
    def fk(self, q: np.ndarray) -> np.ndarray:
        q1, q2 = q[..., 0], q[..., 1]
        return np.stack([self.l1 * np.cos(q1) + self.l2 * np.cos(q1 + q2),
                         self.l1 * np.sin(q1) + self.l2 * np.sin(q1 + q2)], -1)

    def jac(self, q: np.ndarray) -> np.ndarray:
        q1, q2 = q[..., 0], q[..., 1]
        s1, c1 = np.sin(q1), np.cos(q1)
        s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
        return np.stack([
            np.stack([-self.l1 * s1 - self.l2 * s12, -self.l2 * s12], -1),
            np.stack([self.l1 * c1 + self.l2 * c12, self.l2 * c12], -1),
        ], -2)

    # ------------------------------------------------------------- targets
    @property
    def x_star(self) -> np.ndarray:
        return self.fk(np.asarray(self.q_star))

    @property
    def phi_star(self) -> float:
        return float(sum(self.q_star))

    @property
    def q_mirror(self) -> np.ndarray:
        """Elbow-flipped configuration with the same tip position x*
        (reflection of the arm across the base-to-target line)."""
        alpha = np.arctan2(self.x_star[1], self.x_star[0])
        return np.array([2.0 * alpha - self.q_star[0], -self.q_star[1]])

    # ------------------------------------------------------ gradients, Hess
    def grad(self, term: str, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if term == "joint":
            return self.k_q * np.sin(q - np.asarray(self.q_star))
        if term == "position":
            e = self.fk(q) - self.x_star
            return self.k_x * np.einsum("...ij,...i->...j", self.jac(q), e)
        if term == "orientation":
            s = self.k_phi * np.sin(q.sum(-1) - self.phi_star)
            return np.stack([s, s], -1)
        raise ValueError(term)

    def hess(self, term: str, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if term == "joint":
            H = np.zeros(q.shape[:-1] + (2, 2))
            c = self.k_q * np.cos(q - np.asarray(self.q_star))
            H[..., 0, 0], H[..., 1, 1] = c[..., 0], c[..., 1]
            return H
        if term == "position":
            # Hess 1/2||x - x*||^2 = J^T J + sum_k e_k Hess x_k, where the
            # second-derivative term is -(e . a) on (0,0) and -(e . c)
            # everywhere, with a, c the two link vectors
            q1, q2 = q[..., 0], q[..., 1]
            J = self.jac(q)
            e = self.fk(q) - self.x_star
            a = np.stack([self.l1 * np.cos(q1), self.l1 * np.sin(q1)], -1)
            c = np.stack([self.l2 * np.cos(q1 + q2),
                          self.l2 * np.sin(q1 + q2)], -1)
            ea = np.einsum("...i,...i->...", e, a)
            ec = np.einsum("...i,...i->...", e, c)
            H = np.einsum("...ki,...kj->...ij", J, J)
            H[..., 0, 0] -= ea
            H -= ec[..., None, None]
            return self.k_x * H
        if term == "orientation":
            c = self.k_phi * np.cos(q.sum(-1) - self.phi_star)
            return c[..., None, None] * np.ones((2, 2))
        raise ValueError(term)

    def grad_sum(self, terms, q: np.ndarray) -> np.ndarray:
        return sum(self.grad(t, q) for t in terms)

    def hess_sum(self, terms, q: np.ndarray) -> np.ndarray:
        return sum(self.hess(t, q) for t in terms)

    def lambda_min(self, terms, q: np.ndarray) -> np.ndarray:
        """Contraction rate field lambda_min(Hess U)/b (> 0: contracting)."""
        return np.linalg.eigvalsh(self.hess_sum(terms, q))[..., 0] / self.b


def wrap(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def segment_certificate(pot: RodPotentials, terms, Q: np.ndarray,
                        n_s: int = 9) -> float:
    """min of lambda_min(Hess U)/b over every straight (flat-torus) segment
    joining two trajectories at every time. Positive => the max pairwise
    distance is guaranteed to decay at least like e^{-cert t}."""
    N = len(Q)
    worst = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            dq = wrap(Q[i] - Q[j])
            for s in np.linspace(0.0, 1.0, n_s):
                lam = pot.lambda_min(terms, Q[j] + s * dq)
                worst = min(worst, float(lam.min()))
    return worst


def max_pairwise_distance(Q: np.ndarray) -> np.ndarray:
    """Max over pairs of the flat-torus distance ||wrap(q_i - q_j)||."""
    N = len(Q)
    return np.max([np.linalg.norm(wrap(Q[i] - Q[j]), axis=-1)
                   for i in range(N) for j in range(i + 1, N)], axis=0)
