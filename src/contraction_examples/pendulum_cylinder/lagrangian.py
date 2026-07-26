"""Colormap of the pendulum Lagrangian over the tangent bundle TS^1,
with the damped-pendulum state trajectory drawn on top of it.

Per unit mass and rod length (m = 1, so L / (m l^2) has units 1/s^2):

    L(theta, theta_dot) = 1/2 theta_dot^2 + (g/l) cos(theta)

Rendered twice with a shared color scale: as a flat (theta, theta_dot)
heatmap, and painted onto the cylinder embedding of TS^1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from . import cylinder
from .animate import (
    AZIM, ELEV, HALO, INK, MUTED, ORANGE, SURFACE, VEL_SCALE, Z_MAX,
    _facing_factor,
)
from .dynamics import PendulumParams, Trajectory, simulate

# Sequential blue ramp (dataviz reference palette, steps 100 -> 700).
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)


def lagrangian(theta: np.ndarray, theta_dot: np.ndarray, params: PendulumParams) -> np.ndarray:
    return 0.5 * theta_dot**2 + (params.gravity / params.length) * np.cos(theta)


def render_lagrangian(params: PendulumParams, path: Path,
                      traj: Trajectory | None = None, dpi: int = 200) -> None:
    w_max = VEL_SCALE * Z_MAX  # theta_dot at the cylinder rims

    fig = plt.figure(figsize=(12.8, 6.0), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1],
                          left=0.07, right=0.99, top=0.97, bottom=0.11, wspace=0.05)

    # ------------------------------------------------- flat (theta, w) map
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(SURFACE)
    th = np.linspace(-np.pi, np.pi, 361)
    w = np.linspace(-w_max, w_max, 241)
    TH, W = np.meshgrid(th, w)
    LV = lagrangian(TH, W, params)
    norm = Normalize(vmin=LV.min(), vmax=LV.max())
    mesh = ax.pcolormesh(TH, W, LV, cmap=CMAP, norm=norm, shading="gouraud",
                         rasterized=True)
    ax.contour(TH, W, LV, levels=14, colors="white", linewidths=0.6, alpha=0.35)
    if traj is not None:
        # split the wrapped angle at the +-pi seam so no segment spans the jump
        wrapped = np.arctan2(np.sin(traj.theta), np.cos(traj.theta))
        w_tr = traj.theta_dot.astype(float)
        seam = np.where(np.abs(np.diff(wrapped)) > np.pi)[0] + 1
        wrapped = np.insert(wrapped, seam, np.nan)
        w_tr = np.insert(w_tr, seam, np.nan)
        ax.plot(wrapped, w_tr, color=SURFACE, lw=3.4, solid_capstyle="round")
        ax.plot(wrapped, w_tr, color=ORANGE, lw=1.8, solid_capstyle="round")
        ax.plot(wrapped[0], w_tr[0], "o", ms=7, color=ORANGE,
                mec="white", mew=1.2)
        ax.annotate("start", (wrapped[0], w_tr[0]), textcoords="offset points",
                    xytext=(8, 4), fontsize=9, color=ORANGE, path_effects=HALO)

    ax.plot(0, 0, "o", ms=7, color=INK)
    ax.plot([-np.pi, np.pi], [0, 0], "o", ms=7, mfc="none", mec=INK, mew=1.2)
    ax.text(0.10, 0.5, "stable eq.", fontsize=9, color=INK, path_effects=HALO)
    ax.text(-np.pi + 0.10, 0.5, "unstable eq.", fontsize=9, color=INK,
            path_effects=HALO)

    ax.set_xlabel(r"$\theta$  [rad]", fontsize=11, color=INK)
    ax.set_ylabel(r"$\dot{\theta}$  [rad/s]", fontsize=11, color=INK)
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                  [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ax.spines.values():
        s.set_color(MUTED)
        s.set_linewidth(0.8)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.03)
    cbar.set_label(
        r"$\mathcal{L}(\theta,\dot{\theta}) \,/\, m l^2"
        r" = \frac{1}{2}\dot{\theta}^2 + \frac{g}{l}\cos\theta$   [$s^{-2}$]",
        fontsize=10, color=INK,
    )
    cbar.ax.tick_params(colors=MUTED, labelsize=9)
    cbar.outline.set_edgecolor(MUTED)
    cbar.outline.set_linewidth(0.8)

    # ------------------------------------------------- painted cylinder
    ax3 = fig.add_subplot(gs[0, 1], projection="3d", computed_zorder=False)
    ax3.set_facecolor(SURFACE)
    ax3.view_init(elev=ELEV, azim=AZIM)
    ax3.set_box_aspect((1, 1, 1.30))
    ax3.set_xlim(-1.30, 1.30)
    ax3.set_ylim(-1.30, 1.30)
    ax3.set_zlim(-Z_MAX - 0.15, Z_MAX + 0.15)
    ax3.set_axis_off()

    sx, sy, sz = cylinder.surface_mesh(-Z_MAX, Z_MAX, n_theta=241, n_z=161)
    s_theta = np.arctan2(sx, -sy)
    LS = lagrangian(s_theta, sz * VEL_SCALE, params)
    ax3.plot_surface(sx, sy, sz, facecolors=CMAP(norm(LS)), rstride=1, cstride=1,
                     shade=False, antialiased=False, linewidth=0, zorder=1)
    for zc in (-Z_MAX, Z_MAX):
        rx, ry, rz = cylinder.rim_circle(zc)
        ax3.plot(rx, ry, rz, color=MUTED, lw=0.8, alpha=0.6, zorder=2)
    if traj is not None:
        x, y, z = cylinder.embed(traj.theta, traj.theta_dot, VEL_SCALE)
        pts = np.column_stack((x, y, z)).reshape(-1, 1, 3)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        facing = _facing_factor(0.5 * (traj.theta[:-1] + traj.theta[1:]))
        rgba = np.tile(np.append(np.array(to_rgb(ORANGE)), 0.0), (len(segs), 1))
        rgba[:, 3] = 0.95 * facing
        ax3.add_collection3d(
            Line3DCollection(segs, colors=rgba, linewidths=1.9, zorder=3)
        )
        ax3.plot([x[0]], [y[0]], [z[0]], "o", ms=7, color=ORANGE,
                 mec="white", mew=1.2, zorder=4)
        ax3.text(x[0], y[0], z[0] + 0.20, "start", fontsize=9, color=ORANGE,
                 ha="center", zorder=4, path_effects=HALO)

    ax3.plot([0], [-1], [0], "o", ms=7, color=INK, zorder=5)
    ax3.text(-0.30, -1.02, -0.16, "stable eq.", fontsize=9, color=INK,
             ha="right", zorder=5, path_effects=HALO)
    # angle labels on the bottom rim, matching the animation figure
    for th_l, lab in ((0.0, r"$\theta=0$"), (np.pi / 2, r"$\pi/2$"),
                      (np.pi, r"$\pm\pi$"), (-np.pi / 2, r"$-\pi/2$")):
        ax3.text(1.22 * np.sin(th_l), -1.22 * np.cos(th_l), -Z_MAX - 0.22, lab,
                 fontsize=9, color=MUTED, ha="center")
    ax3.text(0, 0, Z_MAX + 0.32, r"$\dot{\theta}$ up the axis", fontsize=9,
             color=MUTED, ha="center")

    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render the Lagrangian colormap on the tangent-bundle cylinder."
    )
    parser.add_argument("--outdir", type=Path, default=Path("media"))
    parser.add_argument("--duration", type=float, default=14.0)
    parser.add_argument("--theta0", type=float, default=-np.pi / 2)
    parser.add_argument("--theta-dot0", type=float, default=11.0)
    parser.add_argument("--damping", type=float, default=0.55)
    parser.add_argument("--no-trajectory", action="store_true",
                        help="render the bare Lagrangian colormap only")
    args = parser.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    params = PendulumParams(damping=args.damping)
    traj = None
    if not args.no_trajectory:
        traj = simulate(params, args.theta0, args.theta_dot0,
                        args.duration, fps=60)
    path = args.outdir / "lagrangian_cylinder.png"
    print(f"Rendering Lagrangian colormap -> {path}")
    render_lagrangian(params, path, traj=traj)
    print("Done.")


if __name__ == "__main__":
    main()
