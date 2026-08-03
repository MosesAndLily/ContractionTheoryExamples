"""The converse (integral) contraction metric: full-metric nonlinear analysis.

With an UNRESTRICTED state-dependent metric, contraction analysis is
complete on the basin of attraction. Define, along the trajectory from x,

    P(x) = int_0^inf e^{2 lam t} Phi(t; x)^T Q Phi(t; x) dt,

where Phi(t; x) = D_x phi_t(x) is the variational (linearized flow)
matrix. Using Phi(t; phi_s(x)) = Phi(t+s; x) Phi(s; x)^{-1}, one gets

    d/ds [ Phi(s)^T P(phi_s x) Phi(s) ] = -2 lam (...) - Phi^T Q Phi,

i.e. pointwise, EXACTLY,

    Pdot + P J + J^T P = -2 lam P - Q   (< -2 lam P),

wherever the integral converges — any compact subset of the basin of the
stable equilibrium, provided lam is below the equilibrium's exponential
rate. So the "full" metric certifies (essentially) the whole basin; the
hard boundary is the basin itself, since contraction is false at the
saddle equilibria. The price: P(x) needs the trajectory from x (it is an
analysis object, not a causal/synthesis one — that is what SOS/CCM
parametrizations are for), and it must be computed numerically.

This module computes P(x) by augmenting the ODE with the variational and
integral states, verifies the exact identity above as a self-test, and
compares the certified guaranteed rate of the three metric families
(constant P0, mechanical P(q), converse P(x)) as the trajectory fan
widens. Entry point: ``uv run converse-contraction``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh

from ..double_pendulum_torus import torus
from ..double_pendulum_torus.animate import AZIM, ELEV
from ..double_pendulum_torus.dynamics import (DoublePendulumParams,
                                              simulate, vector_field)
from ..style import BLUE, GUIDE, HALO, INK, MUTED, ORANGE, SURFACE
from . import metric as metric_mod
from .metric import MechanicalMetric

RED = "#e34948"    # categorical slot 8: the constant metric
AQUA = "#1baf7a"   # categorical slot 3: the converse metric


# ------------------------------------------------------- analytic Jacobian
def analytic_jacobian(p: DoublePendulumParams, x: np.ndarray) -> np.ndarray:
    """Closed-form state Jacobian (verified against finite differences)."""
    t1, t2, w1, w2 = x
    ml = p.m2 * p.l1 * p.l2
    a = (p.m1 + p.m2) * p.l1**2
    c_ = p.m2 * p.l2**2
    delta = t1 - t2
    b = ml * np.cos(delta)
    h = ml * np.sin(delta)
    det = a * c_ - b * b
    g1 = (p.m1 + p.m2) * p.gravity * p.l1
    g2 = p.m2 * p.gravity * p.l2

    r1 = -h * w2**2 - g1 * np.sin(t1) - p.damping1 * w1
    r2 = h * w1**2 - g2 * np.sin(t2) - p.damping2 * w2
    acc1 = (c_ * r1 - b * r2) / det
    acc2 = (a * r2 - b * r1) / det

    # partials of b, h, det w.r.t. (t1, t2): db/dt1 = -h, dh/dt1 = b, ...
    db = np.array([-h, h])
    dh = np.array([b, -b])
    ddet = -2.0 * b * db
    dr1 = np.array([-dh[0] * w2**2 - g1 * np.cos(t1), -dh[1] * w2**2])
    dr2 = np.array([dh[0] * w1**2, dh[1] * w1**2 - g2 * np.cos(t2)])

    J = np.zeros((4, 4))
    J[0, 2] = J[1, 3] = 1.0
    for k in range(2):  # d acc / d theta_k
        J[2, k] = ((c_ * dr1[k] - db[k] * r2 - b * dr2[k]) - acc1 * ddet[k]) / det
        J[3, k] = ((a * dr2[k] - db[k] * r1 - b * dr1[k]) - acc2 * ddet[k]) / det
    # d acc / d w
    dr1w = np.array([-p.damping1, -2.0 * h * w2])
    dr2w = np.array([2.0 * h * w1, -p.damping2])
    for k in range(2):
        J[2, 2 + k] = (c_ * dr1w[k] - b * dr2w[k]) / det
        J[3, 2 + k] = (a * dr2w[k] - b * dr1w[k]) / det
    return J


# ------------------------------------------------------- converse metric
def variational_P(p: DoublePendulumParams, x0: np.ndarray, lam: float = 0.1,
                  T: float = 40.0, rtol: float = 1e-8
                  ) -> tuple[np.ndarray, float]:
    """P(x0) = int_0^T e^{2 lam t} Phi^T Phi dt (Q = I) and the final
    state distance to the equilibrium (a basin/convergence check)."""

    def rhs(t, y):
        x, Phi = y[:4], y[4:20].reshape(4, 4)
        J = analytic_jacobian(p, x)
        dS = np.exp(2.0 * lam * t) * Phi.T @ Phi
        return np.concatenate([
            vector_field(t, x, p), (J @ Phi).ravel(), dS.ravel(),
        ])

    y0 = np.concatenate([x0, np.eye(4).ravel(), np.zeros(16)])
    sol = solve_ivp(rhs, (0.0, T), y0, method="DOP853", rtol=rtol, atol=1e-10)
    xf = sol.y[:4, -1]
    resid = float(np.hypot(
        np.linalg.norm(np.arctan2(np.sin(xf[:2]), np.cos(xf[:2]))),
        np.linalg.norm(xf[2:]),
    ))
    S = sol.y[20:36, -1].reshape(4, 4)
    return 0.5 * (S + S.T), resid


def converse_rate_matrix(p: DoublePendulumParams, x: np.ndarray,
                         lam: float = 0.1, T: float = 40.0,
                         h: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """(Pdot + PJ + J^T P, P) for the converse metric, Pdot by a
    directional finite difference along the flow."""
    f = np.array(vector_field(0.0, x, p))
    scale = h / max(1.0, np.linalg.norm(f))
    Pp, _ = variational_P(p, x + scale * f, lam, T)
    Pm, _ = variational_P(p, x - scale * f, lam, T)
    P, _ = variational_P(p, x, lam, T)
    Pdot = (Pp - Pm) / (2.0 * scale)
    J = analytic_jacobian(p, x)
    return Pdot + P @ J + J.T @ P, P


# ------------------------------------------------------------- certificates
def segment_points(Z: np.ndarray, t_stride: int, n_interp: int) -> np.ndarray:
    """Sampled states on straight segments between all trajectory pairs."""
    N = Z.shape[0]
    pts = []
    svals = np.linspace(0.0, 1.0, n_interp)
    for ti in range(0, Z.shape[1], t_stride):
        states = Z[:, ti]
        for i in range(N):
            for j in range(i + 1, N):
                for s in svals:
                    pts.append((1.0 - s) * states[i] + s * states[j])
    return np.array(pts)


def guaranteed_rate(S_and_P) -> float:
    """min over points of the signed generalized-eigenvalue rate (against
    the metric); negative exactly where the condition fails."""
    rate = np.inf
    for S, P in S_and_P:
        rate = min(rate, eigh(-S, P, eigvals_only=True)[0])
    return 0.5 * rate


def scan(p: DoublePendulumParams, eps: float, lam: float, spreads,
         duration: float, t_stride: int, n_interp: int):
    """Guaranteed rate vs fan spread for the three metric families."""
    mech = MechanicalMetric(p, eps)
    const = MechanicalMetric(p, eps, constant=True)
    rows = []
    for spread in spreads:
        q0s = [(a, b) for a in (-spread, 0.0, spread)
               for b in (-spread, 0.0, spread)]
        trajs = [simulate(p, a, b, 0.0, 0.0, duration, 60) for a, b in q0s]
        Z = np.stack([np.column_stack((t.theta1, t.theta2,
                                       t.theta1_dot, t.theta2_dot))
                      for t in trajs])
        pts = segment_points(Z, t_stride, n_interp)
        r_const = guaranteed_rate(const.rate_matrix(x) for x in pts)
        r_mech = guaranteed_rate(mech.rate_matrix(x) for x in pts)
        # converse metric: also watch basin convergence
        _, resid = variational_P(p, np.array([spread, spread, 0, 0]), lam)
        if resid > 0.2:
            r_conv = np.nan  # fan corner leaves the basin: metric undefined
        else:
            r_conv = guaranteed_rate(
                converse_rate_matrix(p, x, lam) for x in pts)
        rows.append((spread, r_const, r_mech, r_conv))
        print(f"  spread {spread:4.2f}: constant {r_const:+.3f}  "
              f"mechanical {r_mech:+.3f}  converse "
              f"{'diverged' if np.isnan(r_conv) else f'{r_conv:+.3f}'}")
    return np.array(rows)


# ------------------------------------------------------------------ figure
def render_figure(p: DoublePendulumParams, rows: np.ndarray, lam: float,
                  eps: float, fan_spread: float, duration: float,
                  path: Path, dpi: int = 200) -> None:
    fig = plt.figure(figsize=(12.8, 6.0), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1],
                          left=0.06, right=0.99, top=0.90, bottom=0.12,
                          wspace=0.14)

    # ------------------------------------------- left: rate vs fan spread
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.axhline(0.0, color=GUIDE, lw=1.0)
    ax.set_xlabel("fan spread  [rad]", fontsize=10, color=INK)
    ax.set_ylabel("guaranteed contraction rate on segments  [1/s]",
                  fontsize=10, color=INK)
    s = rows[:, 0]
    ax.plot(s, rows[:, 1], marker="o", ls=(0, (5, 3)), lw=1.8, ms=5,
            color=RED, label=r"constant $P_0$")
    ax.plot(s, rows[:, 2], marker="o", lw=1.8, ms=5, color=BLUE,
            label=r"mechanical $P(q)$")
    ax.plot(s, rows[:, 3], marker="o", lw=1.8, ms=5, color=AQUA,
            label=r"converse $P(x)=\int e^{2\lambda t}\Phi^{\top}\!\Phi\,dt$"
                  rf"  ($\lambda={lam:g}$)")
    ax.text(s[-1], 0.012, "certified  ↑   /   ↓  fails", ha="right",
            fontsize=8.5, color=MUTED)
    ax.legend(loc="lower left", fontsize=9, frameon=False, labelcolor=INK)

    # ------------------------- right: wide fan on the torus, wireframe view
    ax3 = fig.add_subplot(gs[0, 1], projection="3d", computed_zorder=False)
    ax3.set_facecolor(SURFACE)
    ax3.view_init(elev=ELEV, azim=AZIM)
    ax3.set_xlim(-1.55, 1.55)
    ax3.set_ylim(-1.55, 1.55)
    ax3.set_zlim(-0.62, 0.62)
    ax3.set_box_aspect((1, 1, 0.40))
    ax3.set_axis_off()
    for th1 in np.arange(-np.pi, np.pi, np.pi / 12):
        mx, my, mz = torus.meridian(th1)
        ax3.plot(mx, my, mz, color=GUIDE, lw=0.6, alpha=0.45, zorder=1)
    for th2 in np.arange(-np.pi, np.pi, np.pi / 6):
        px, py, pz = torus.parallel_circle(th2)
        ax3.plot(px, py, pz, color=GUIDE, lw=0.6, alpha=0.45, zorder=1)

    # zero contour of the MECHANICAL metric's slice field, for contrast
    from contourpy import contour_generator
    T1, T2, LAM = metric_mod.slice_field(MechanicalMetric(p, eps), n=41)
    cg = contour_generator(x=T1, y=T2, z=LAM)
    for line in cg.lines(0.0):
        t1l, t2l = line[:, 0], line[:, 1]
        x, y, z = torus.embed(t1l, t2l)
        nx, ny, nz = torus.normal(t1l, t2l)
        ax3.plot(x + 0.012 * nx, y + 0.012 * ny, z + 0.012 * nz,
                 color=INK, lw=1.1, ls=(0, (4, 3)), zorder=2)

    q0s = [(a, b) for a in (-fan_spread, 0.0, fan_spread)
           for b in (-fan_spread, 0.0, fan_spread)]
    for a, b in q0s:
        tr = simulate(p, a, b, 0.0, 0.0, duration, 120)
        x, y, z = torus.embed(tr.theta1, tr.theta2)
        nx, ny, nz = torus.normal(tr.theta1, tr.theta2)
        ax3.plot(x + 0.015 * nx, y + 0.015 * ny, z + 0.015 * nz,
                 color=ORANGE, lw=1.2, alpha=0.75, zorder=5)
        ax3.plot([x[0]], [y[0]], [z[0]], "o", ms=6, mfc="none", mec=ORANGE,
                 mew=1.2, alpha=0.8, zorder=6)
    ex, ey, ez = torus.embed(np.array([0.0]), np.array([0.0]))
    ax3.plot([ex[0]], [ey[0]], [ez[0] + 0.02], "o", ms=7, color=INK, zorder=6)
    ax3.text(ex[0] - 0.40, ey[0], ez[0] - 0.16, "stable eq.", fontsize=9,
             color=INK, ha="right", zorder=6, path_effects=HALO)
    ax3.text2D(
        0.5, 0.99,
        rf"fan at $\pm{fan_spread:g}$ rad — far outside the mechanical"
        " region (dashed) —\nstill certified by the converse metric",
        transform=ax3.transAxes, fontsize=9, color=MUTED, ha="center",
        va="top", path_effects=HALO,
    )

    fig.text(0.5, 0.035,
             r"converse metric identity: $\dot P + PJ + J^{\top}\!P"
             r" = -2\lambda P - Q$ exactly, wherever "
             r"$P(x)=\int_0^{\infty} e^{2\lambda t}\Phi^{\top}Q\,\Phi\,dt$"
             " converges (the basin)",
             ha="center", fontsize=9.5, color=MUTED)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Full-metric (converse) nonlinear contraction analysis "
                    "of the damped double pendulum."
    )
    parser.add_argument("--outdir", type=Path, default=Path("media"))
    parser.add_argument("--gravity", type=float, default=2.0)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=0.25)
    parser.add_argument("--lam", type=float, default=0.1,
                        help="target rate of the converse metric; must be "
                             "below the equilibrium rate (~0.22 here)")
    parser.add_argument("--spreads", type=float, nargs="+",
                        default=[0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05])
    parser.add_argument("--fan-spread", type=float, default=0.9,
                        help="fan drawn on the torus panel")
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--t-stride", type=int, default=120,
                        help="segment sampling stride (60 samples/s base)")
    parser.add_argument("--n-interp", type=int, default=3)
    args = parser.parse_args(argv)

    p = DoublePendulumParams(gravity=args.gravity, m1=1.0, m2=1.0,
                             l1=1.0, l2=1.0,
                             damping1=args.damping, damping2=args.damping)

    # self-test 1: analytic Jacobian against finite differences
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.uniform(-2, 2, 4)
        assert np.allclose(analytic_jacobian(p, x),
                           metric_mod.jacobian(p, x), atol=1e-5), \
            "analytic Jacobian disagrees with finite differences"
    print("self-test: analytic Jacobian == finite differences  OK")

    # self-test 2: the exact converse identity R = -2 lam P - Q
    x = np.array([0.6, -0.4, 0.3, -0.2])
    S, P = converse_rate_matrix(p, x, args.lam)
    err = np.linalg.norm(S + 2.0 * args.lam * P + np.eye(4)) / np.linalg.norm(P)
    print(f"self-test: ||R + 2 lam P + Q|| / ||P|| = {err:.2e}  "
          f"({'OK' if err < 5e-3 else 'LARGE — check T / tolerances'})")

    print("Scanning guaranteed rate vs fan spread (three metrics)...")
    rows = scan(p, args.eps, args.lam, args.spreads, args.duration,
                args.t_stride, args.n_interp)

    args.outdir.mkdir(parents=True, exist_ok=True)
    png = args.outdir / "converse_contraction.png"
    print(f"Rendering figure -> {png}")
    render_figure(p, rows, args.lam, args.eps, args.fan_spread,
                  args.duration, png)
    print("Done.")


if __name__ == "__main__":
    main()
