"""MuJoCo model of N overlaid planar two-rod arms, and the first-order flow
b q_dot = -grad U(q) evaluated through MuJoCo's kinematics.

MuJoCo supplies the forward kinematics (tip site position and frame) and
the site Jacobians; the task-space gradients are the usual impedance
pull-backs J_x^T k_x (x - x*) and J_phi^T k_phi sin(phi - phi*). The flow is
first order (no inertia), so it is integrated with classical RK4 on qpos
rather than with mj_step. The same model renders every frame.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .potentials import RodPotentials

ARM_RGBA = "0.922 0.408 0.204 0.55"    # style.ORANGE, translucent
GHOST_RGBA = "0.32 0.32 0.31 0.30"     # style.MUTED: the target arm q*
TARGET_RGBA = "0.165 0.471 0.839 1"    # style.BLUE: x* and phi*


def build_xml(pot: RodPotentials, n_arms: int) -> str:
    """MJCF: n_arms independent arms at the origin (hinges about z) plus
    static markers for the targets q* (ghost arm), x* (dot), phi* (arrow)."""
    l1, l2 = pot.l1, pot.l2
    r = 0.045
    arms = "".join(f"""
    <body name="link1_{k}" pos="0 0 {0.002 * k:.3f}">
      <joint name="q1_{k}" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 {l1} 0 0" size="{r}" rgba="{ARM_RGBA}"/>
      <body name="link2_{k}" pos="{l1} 0 0">
        <joint name="q2_{k}" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 {l2} 0 0" size="{r}" rgba="{ARM_RGBA}"/>
        <site name="tip_{k}" pos="{l2} 0 0" size="0.02" rgba="0 0 0 0"/>
      </body>
    </body>""" for k in range(n_arms))

    e1 = l1 * np.array([np.cos(pot.q_star[0]), np.sin(pot.q_star[0])])
    xs = pot.x_star
    head = xs + 0.45 * np.array([np.cos(pot.phi_star), np.sin(pot.phi_star)])
    reach = 1.15 * (l1 + l2)
    return f"""
<mujoco model="planar_two_rod">
  <option gravity="0 0 0"/>
  <visual>
    <headlight ambient="0.6 0.6 0.6" diffuse="0.4 0.4 0.4" specular="0 0 0"/>
    <global offwidth="1200" offheight="1200"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.988 0.988 0.984"
             rgb2="0.945 0.945 0.935" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0"/>
  </asset>
  <worldbody>
    <geom type="plane" size="{2 * reach:.3f} {2 * reach:.3f} 0.1" pos="0 0 -0.1" material="grid"/>
    <camera name="top" pos="0 0 {reach / np.tan(np.deg2rad(20)):.3f}" xyaxes="1 0 0 0 1 0" fovy="40"/>
    <geom type="cylinder" size="0.09 0.05" pos="0 0 -0.05" rgba="0.043 0.043 0.043 1"/>
    <geom type="capsule" fromto="0 0 -0.04 {e1[0]:.4f} {e1[1]:.4f} -0.04"
          size="0.05" rgba="{GHOST_RGBA}"/>
    <geom type="capsule" fromto="{e1[0]:.4f} {e1[1]:.4f} -0.04 {xs[0]:.4f} {xs[1]:.4f} -0.04"
          size="0.05" rgba="{GHOST_RGBA}"/>
    <geom type="sphere" size="0.085" pos="{xs[0]:.4f} {xs[1]:.4f} 0.1" rgba="{TARGET_RGBA}"/>
    <geom type="capsule" fromto="{xs[0]:.4f} {xs[1]:.4f} 0.1 {head[0]:.4f} {head[1]:.4f} 0.1"
          size="0.025" rgba="{TARGET_RGBA}"/>{arms}
  </worldbody>
</mujoco>"""


class RodArms:
    """N overlaid arms sharing one MuJoCo model; ``q`` has shape (N, 2)."""

    def __init__(self, pot: RodPotentials, n_arms: int):
        self.pot = pot
        self.n = n_arms
        self.model = mujoco.MjModel.from_xml_string(build_xml(pot, n_arms))
        self.data = mujoco.MjData(self.model)
        self.tips = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                       f"tip_{k}") for k in range(n_arms)]
        self._jp = np.zeros((3, self.model.nv))
        self._jr = np.zeros((3, self.model.nv))

    def set_q(self, q: np.ndarray) -> None:
        self.data.qpos[:] = np.asarray(q).reshape(-1)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def pose_for_render(self, q: np.ndarray) -> None:
        """set_q plus camera/light placement (not done by mj_kinematics)."""
        self.set_q(q)
        mujoco.mj_camlight(self.model, self.data)

    def grad(self, term: str, q: np.ndarray) -> np.ndarray:
        """grad U_term at every arm's configuration, via MuJoCo."""
        pot = self.pot
        self.set_q(q)
        g = np.zeros((self.n, 2))
        for k, sid in enumerate(self.tips):
            dofs = slice(2 * k, 2 * k + 2)
            if term == "joint":
                g[k] = pot.k_q * np.sin(self.data.qpos[dofs]
                                        - np.asarray(pot.q_star))
                continue
            mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, sid)
            if term == "position":
                e = self.data.site_xpos[sid, :2] - pot.x_star
                g[k] = pot.k_x * self._jp[:2, dofs].T @ e
            elif term == "orientation":
                R = self.data.site_xmat[sid].reshape(3, 3)
                phi = np.arctan2(R[1, 0], R[0, 0])
                g[k] = (pot.k_phi * np.sin(phi - pot.phi_star)
                        * self._jr[2, dofs])
            else:
                raise ValueError(term)
        return g

    def velocity(self, terms, q: np.ndarray) -> np.ndarray:
        return -sum(self.grad(t, q) for t in terms) / self.pot.b

    def simulate(self, terms, q0: np.ndarray, duration: float,
                 n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """RK4 on q_dot = -grad U / b. Returns t (T,) and Q (N, T, 2)."""
        t = np.linspace(0.0, duration, n_samples + 1)
        dt = t[1] - t[0]
        q = np.array(q0, dtype=float)
        Q = np.empty((len(t),) + q.shape)
        Q[0] = q
        for i in range(1, len(t)):
            k1 = self.velocity(terms, q)
            k2 = self.velocity(terms, q + 0.5 * dt * k1)
            k3 = self.velocity(terms, q + 0.5 * dt * k2)
            k4 = self.velocity(terms, q + dt * k3)
            q = q + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            Q[i] = q
        return t, np.swapaxes(Q, 0, 1)

    def self_test(self, rng: np.random.Generator) -> float:
        """Max |MuJoCo gradient - closed-form gradient| over random q."""
        q = rng.uniform(-np.pi, np.pi, size=(self.n, 2))
        return max(float(np.abs(self.grad(t, q) - self.pot.grad(t, q)).max())
                   for t in ("joint", "position", "orientation"))
