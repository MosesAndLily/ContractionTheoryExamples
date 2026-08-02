"""Contraction of the damped double pendulum on its configuration torus.

Left: the torus painted with lambda_max(PJ + J^T P) on the q_dot = 0
slice (blue: contracting, red: expanding, ink dashed: the zero contour),
with a 3x3 fan of configuration trajectories converging to the hanging
equilibrium inside the contracting region.

Right: max pairwise distance between the nine trajectories — Euclidean vs
the mechanical block metric P (mass matrix as velocity block + stiffness
+ cross term), mirroring the earlier examples.

Entry point: ``uv run double-pendulum-contraction``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from contourpy import contour_generator
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from ..double_pendulum_torus import torus
from ..double_pendulum_torus.animate import AZIM, ELEV, _facing_factor
from ..double_pendulum_torus.dynamics import (DoublePendulumParams,
                                              Trajectory, simulate)
from ..mass_spring_damper.metrics import quad_form
from ..media_utils import mp4_to_gif, render_video
from ..style import BLUE, GUIDE, HALO, INK, MUTED, ORANGE, SURFACE
from .metric import (damping_matrix, jacobian, mass_matrix_at_origin,
                     segment_certificate, slice_field, stiffness_at_origin,
                     structured_metric)

RED = "#e34948"    # categorical slot 8: expanding / the Euclidean metric
OVERSAMPLE = 4     # simulation samples per video frame
DIVERGING = LinearSegmentedColormap.from_list(
    "contraction_div", [BLUE, "#f0efec", RED]
)
LAM_CAP = 8.0      # clip the expanding side of the slice colormap


def wrap(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


class DoublePendulumContractionFigure:
    """Builds the two-panel figure and exposes ``update(frame)`` for animation."""

    def __init__(self, trajs: list, params: DoublePendulumParams,
                 P: np.ndarray, eps: float, slice_data, cert, dpi: int = 100):
        self.params = params
        self.P = P
        self.eps = eps
        self.t = trajs[0].t
        self.n = len(self.t)
        self.n_frames = int(np.ceil(self.n / OVERSAMPLE))
        self.N = len(trajs)
        self.Z = np.stack([np.column_stack((tr.theta1, tr.theta2,
                                            tr.theta1_dot, tr.theta2_dot))
                           for tr in trajs])
        self.d_eye = self._max_pairwise(np.eye(4))
        self.d_p = self._max_pairwise(P)

        # per-trajectory 3D points (pushed slightly off the painted surface)
        self.pts3d, self.facing = [], []
        for i in range(self.N):
            t1, t2 = self.Z[i, :, 0], self.Z[i, :, 1]
            x, y, z = torus.embed(t1, t2)
            nx, ny, nz = torus.normal(t1, t2)
            self.pts3d.append(np.column_stack((x + 0.015 * nx, y + 0.015 * ny,
                                               z + 0.015 * nz)))
            self.facing.append(_facing_factor(0.5 * (t1[:-1] + t1[1:]),
                                              0.5 * (t2[:-1] + t2[1:])))
        self.orange_rgb = np.array(to_rgb(ORANGE))

        self.fig = plt.figure(figsize=(12.8, 6.0), dpi=dpi)
        self.fig.patch.set_facecolor(SURFACE)
        gs = self.fig.add_gridspec(
            1, 2, width_ratios=[1.15, 1.0],
            left=0.005, right=0.985, top=0.97, bottom=0.11, wspace=0.10,
        )
        self.time_text = self.fig.text(
            0.02, 0.965, "", ha="left", va="center",
            fontsize=11, color=MUTED, family="monospace",
        )
        self._build_left(gs[0, 0], slice_data, cert)
        self._build_right(gs[0, 1])

    def _max_pairwise(self, M: np.ndarray) -> np.ndarray:
        dists = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                dz = self.Z[i] - self.Z[j]
                dz[:, :2] = wrap(dz[:, :2])
                dists.append(np.sqrt(quad_form(dz, M)))
        return np.max(dists, axis=0)

    # ------------------------------------------------------------------ left
    def _build_left(self, spec, slice_data, cert) -> None:
        ax = self.fig.add_subplot(spec, projection="3d", computed_zorder=False)
        self.ax_left = ax
        ax.set_facecolor(SURFACE)
        ax.view_init(elev=ELEV, azim=AZIM)
        ax.set_xlim(-1.55, 1.55)
        ax.set_ylim(-1.55, 1.55)
        ax.set_zlim(-0.62, 0.62)
        ax.set_box_aspect((1, 1, 0.40))
        ax.set_axis_off()

        # painted q_dot = 0 slice: lambda_max(PJ + J^T P) on the torus
        T1, T2, LAM = slice_data
        sx, sy, sz = torus.embed(T1, T2)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=LAM.min(), vmax=LAM_CAP)
        colors = DIVERGING(norm(np.clip(LAM, None, LAM_CAP)))
        ax.plot_surface(sx, sy, sz, facecolors=colors, rstride=1, cstride=1,
                        shade=False, antialiased=False, linewidth=0, zorder=1)
        # zero contour of the slice field, pushed off the surface
        cg = contour_generator(x=T1, y=T2, z=LAM)
        for line in cg.lines(0.0):
            t1, t2 = line[:, 0], line[:, 1]
            x, y, z = torus.embed(t1, t2)
            nx, ny, nz = torus.normal(t1, t2)
            ax.plot(x + 0.012 * nx, y + 0.012 * ny, z + 0.012 * nz,
                    color=INK, lw=1.1, ls=(0, (4, 3)), zorder=2)

        cax = self.fig.add_axes([0.045, 0.085, 0.26, 0.022])
        cbar = self.fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=DIVERGING), cax=cax,
            orientation="horizontal", extend="max",
        )
        cbar.set_label(
            r"$\lambda_{\max}(PJ + J^{\top}\!P)$ on the $\dot{q}=0$ slice"
            r"   ($\prec 0$: contracting)",
            fontsize=8.5, color=INK,
        )
        cbar.set_ticks([round(float(LAM.min()), 1), 0, 2, 4, 6, 8])
        cbar.ax.tick_params(colors=MUTED, labelsize=8)
        cbar.outline.set_edgecolor(MUTED)
        cbar.outline.set_linewidth(0.8)

        # equilibria: stable at (0,0), the other three are saddles/sources
        ex, ey, ez = torus.embed(np.array([0.0]), np.array([0.0]))
        ax.plot([ex[0]], [ey[0]], [ez[0] + 0.02], "o", ms=7, color=INK,
                zorder=6)
        ax.text(ex[0] - 0.40, ey[0], ez[0] - 0.16, "stable eq.", fontsize=9,
                color=INK, ha="right", zorder=6, path_effects=HALO)
        for t1e, t2e in ((0.0, np.pi), (np.pi, 0.0), (np.pi, np.pi)):
            ex, ey, ez = torus.embed(np.array([t1e]), np.array([t2e]))
            ax.plot([ex[0]], [ey[0]], [ez[0]], "o", ms=6, mfc="none",
                    mec=MUTED, mew=1.1, alpha=0.7, zorder=2)

        # ghosts of the initial fan and the trajectory artists
        for i in range(self.N):
            ax.plot([self.pts3d[i][0, 0]], [self.pts3d[i][0, 1]],
                    [self.pts3d[i][0, 2]], "o", ms=6, mfc="none", mec=ORANGE,
                    mew=1.2, alpha=0.7, zorder=3)
        self.traces = []
        for i in range(self.N):
            # seeded with a degenerate segment: add_collection3d cannot
            # autoscale from an empty collection
            seed = [np.stack([self.pts3d[i][0], self.pts3d[i][0]])]
            lc = Line3DCollection(seed, linewidths=1.5, zorder=5)
            ax.add_collection3d(lc)
            self.traces.append(lc)
        self.dots = [ax.plot([], [], [], "o", ms=7, color=ORANGE, mec="white",
                             mew=1.2, zorder=7)[0] for _ in range(self.N)]

        M0 = mass_matrix_at_origin(self.params)
        K0 = stiffness_at_origin(self.params)
        p = self.params
        ax.text2D(
            0.0, 0.985,
            f"M(q)q̈ + C(q,q̇)q̇ + g(q) + Dq̇ = 0   "
            f"(m={p.m1:g}, l={p.l1:g}, g={p.gravity:g}, D={p.damping1:g}·I)\n"
            f"M₀ = [ {M0[0, 0]:g} {M0[0, 1]:g} ; {M0[1, 0]:g} {M0[1, 1]:g} ]"
            f"   K₀ = diag({K0[0, 0]:g}, {K0[1, 1]:g})   ε = {self.eps:g}\n"
            "P = [ K₀+εD  εM₀ ; εM₀  M₀ ]  →  "
            "PA+AᵀP = −2·blkdiag(εK₀, D−εM₀) ≺ 0\n"
            "ε=0 (energy metric): λmax(PA+AᵀP) = 0 — only semi-contraction\n"
            f"certificate on inter-trajectory segments: "
            f"max λmax(PJ+JᵀP) = {cert[0]:+.2f} ≺ 0",
            transform=ax.transAxes, fontsize=8, color=MUTED,
            family="monospace", va="top", path_effects=HALO,
        )

    # ----------------------------------------------------------------- right
    def _build_right(self, spec) -> None:
        ax = self.fig.add_subplot(spec)
        self.ax_right = ax
        ax.set_facecolor(SURFACE)
        ax.set_yscale("log")
        ax.set_xlim(0.0, self.t[-1])
        ax.set_ylim(self.d_p[-1] * 0.35, self.d_p[0] * 4.0)
        ax.set_xlabel("t  [s]", fontsize=10, color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelsize=9)

        # average rate: slowest mode of the linearization
        M0 = mass_matrix_at_origin(self.params)
        K0 = stiffness_at_origin(self.params)
        D = damping_matrix(self.params)
        A = np.block([[np.zeros((2, 2)), np.eye(2)],
                      [-np.linalg.solve(M0, K0), -np.linalg.solve(M0, D)]])
        rate = -np.linalg.eigvals(A).real.max()
        ref = self.d_p[0] * np.exp(-rate * self.t)
        ax.plot(self.t, ref, color=GUIDE, lw=1.0, zorder=2)
        ax.text(self.t[-1] * 0.72, ref[int(0.72 * self.n)] * 0.42,
                rf"average: $\propto e^{{-{rate:.2f}\,t}}$", fontsize=9,
                color=MUTED)
        self.rate_bound = None  # set via set_bound() once certified

        (self.line_eye,) = ax.plot(
            [], [], ls=(0, (5, 3)), lw=1.8, color=RED, zorder=3,
            label=r"$\max_{i,j}\ \|\Delta\mathbf{x}_{ij}\|$ — not monotone (flats, even growth)",
        )
        (self.line_p,) = ax.plot(
            [], [], lw=1.8, color=BLUE, zorder=4,
            label=r"$\max_{i,j}\ \sqrt{\Delta\mathbf{x}_{ij}^{\top} P\, \Delta\mathbf{x}_{ij}}$ — monotone, exponential: contraction",
        )
        ax.text(0.03, 0.04,
                r"$\Delta\mathbf{x}_{ij} = (\Delta q$ on $T^2,\ \Delta\dot{q})$",
                transform=ax.transAxes, fontsize=9, color=MUTED)
        ax.legend(loc="upper right", fontsize=8, frameon=False,
                  labelcolor=INK, handlelength=1.6)

    def set_bound(self, rate_bound: float) -> None:
        """Draw the guaranteed envelope from the segment certificate."""
        if rate_bound <= 0:
            return
        ref_g = self.d_p[0] * np.exp(-rate_bound * self.t)
        self.ax_right.plot(self.t, ref_g, color=MUTED, lw=1.0, ls=(0, (6, 3)),
                           alpha=0.65, zorder=2)
        self.ax_right.text(
            self.t[-1] * 0.52, ref_g[int(0.52 * self.n)] * 0.55,
            rf"guaranteed bound (segments): $\propto e^{{-{rate_bound:.2f}\,t}}$",
            fontsize=9, color=MUTED, ha="center",
        )

    # ---------------------------------------------------------------- update
    def _place_state(self, i: int) -> None:
        for k in range(self.N):
            x, y, z = self.pts3d[k][i]
            self.dots[k].set_data_3d([x], [y], [z])
        self.time_text.set_text(f"t = {self.t[i]:5.2f} s")

    def _set_traces(self, i: int) -> None:
        for k in range(self.N):
            pts = self.pts3d[k][: i + 1].reshape(-1, 1, 3)
            if len(pts) < 2:
                self.traces[k].set_segments([])
                continue
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            rgba = np.tile(np.append(self.orange_rgb, 0.0), (len(segs), 1))
            rgba[:, 3] = 0.85 * self.facing[k][: len(segs)]
            self.traces[k].set_segments(segs)
            self.traces[k].set_color(rgba)

    def update(self, frame: int):
        """Animation callback: fan and distance curves at video ``frame``."""
        i = min(frame * OVERSAMPLE, self.n - 1)
        self._place_state(i)
        self._set_traces(i)
        self.line_eye.set_data(self.t[: i + 1], self.d_eye[: i + 1])
        self.line_p.set_data(self.t[: i + 1], self.d_p[: i + 1])
        return ()

    def draw_summary(self, i_snap: int) -> None:
        """Full traces and curves with the fan pose at ``i_snap``."""
        self._place_state(i_snap)
        self._set_traces(self.n - 1)
        self.line_eye.set_data(self.t, self.d_eye)
        self.line_p.set_data(self.t, self.d_p)


# ----------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render the double-pendulum contraction-on-the-torus "
                    "analysis and figures."
    )
    parser.add_argument("--outdir", type=Path, default=Path("media"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--gravity", type=float, default=2.0)
    parser.add_argument("--damping", type=float, default=1.0,
                        help="joint damping (D = damping * I)")
    parser.add_argument("--eps", type=float, default=0.3,
                        help="cross-term weight of the block metric")
    parser.add_argument("--spread", type=float, default=0.15,
                        help="fan is the 3x3 grid of (theta1, theta2) in "
                             "[-spread, +spread]^2, released from rest; the "
                             "default is inside the certified region for the "
                             "default eps and damping")
    parser.add_argument("--t-snap", type=float, default=2.0,
                        help="time of the fan pose in the summary PNG")
    parser.add_argument("--no-video", action="store_true",
                        help="render only the summary PNG")
    args = parser.parse_args(argv)
    if args.duration * args.fps < 2:
        parser.error("duration must cover at least two frames (duration >= 2/fps)")

    params = DoublePendulumParams(gravity=args.gravity, m1=1.0, m2=1.0,
                                  l1=1.0, l2=1.0,
                                  damping1=args.damping, damping2=args.damping)
    P = structured_metric(params, args.eps)

    q0s = [(a, b) for a in (-args.spread, 0.0, args.spread)
           for b in (-args.spread, 0.0, args.spread)]
    trajs = [simulate(params, a, b, 0.0, 0.0, args.duration,
                      args.fps * OVERSAMPLE) for a, b in q0s]
    Z = np.stack([np.column_stack((tr.theta1, tr.theta2,
                                   tr.theta1_dot, tr.theta2_dot))
                  for tr in trajs])

    print("Checking the contraction certificate on inter-trajectory segments...")
    cert = segment_certificate(params, P, Z)
    if cert[0] >= 0:
        print(f"warning: certificate FAILS (max lambda_max = {cert[0]:+.3f}) — "
              "reduce --spread or increase --damping; the P-distance may "
              "not decay monotonically")
    else:
        print(f"  certified: max lambda_max = {cert[0]:+.3f}, "
              f"guaranteed distance rate {cert[1]:.3f}")
    print("Painting lambda_max(PJ + J^T P) on the q_dot = 0 slice...")
    slice_data = slice_field(params, P, n=71)

    args.outdir.mkdir(parents=True, exist_ok=True)
    png = args.outdir / "double_pendulum_contraction.png"
    print(f"Rendering summary image -> {png}")
    fig_obj = DoublePendulumContractionFigure(trajs, params, P, args.eps,
                                              slice_data, cert, dpi=200)
    if cert[0] < 0:
        fig_obj.set_bound(cert[1])
    i_snap = min(int(np.searchsorted(fig_obj.t, args.t_snap)), fig_obj.n - 1)
    fig_obj.draw_summary(i_snap)
    fig_obj.fig.savefig(png, dpi=200, facecolor=SURFACE)
    plt.close(fig_obj.fig)

    if not args.no_video:
        mp4 = args.outdir / "double_pendulum_contraction.mp4"
        gif = args.outdir / "double_pendulum_contraction.gif"
        print(f"Rendering video -> {mp4}")
        fig_obj = DoublePendulumContractionFigure(trajs, params, P, args.eps,
                                                  slice_data, cert)
        if cert[0] < 0:
            fig_obj.set_bound(cert[1])
        render_video(fig_obj.fig, fig_obj.update, fig_obj.n_frames, mp4,
                     fps=args.fps)
        plt.close(fig_obj.fig)
        print(f"Rendering GIF preview -> {gif}")
        mp4_to_gif(mp4, gif)
    print("Done.")


if __name__ == "__main__":
    main()
