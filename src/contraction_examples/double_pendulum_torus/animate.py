"""Two-panel figure: a double pendulum (left) and its configuration
(theta1, theta2) tracing a wireframe torus T^2 = S^1 x S^1 (right).

Renders an MP4 video, a GIF preview, and a summary PNG into an output
directory. Entry point: ``uv run double-pendulum-torus``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from ..media_utils import mp4_to_gif, render_video
from ..style import BLUE, GUIDE, HALO, INK, MUTED, ORANGE, SURFACE
from . import torus
from .dynamics import DoublePendulumParams, Trajectory, simulate

TRAIL_FRAMES = 45  # length of the bright "recent" window, in video frames
OVERSAMPLE = 4     # simulation samples per video frame, for a smooth trace
ELEV, AZIM = 40.0, -78.0


def _facing_factor(theta1: np.ndarray, theta2: np.ndarray) -> np.ndarray:
    """How much the torus surface at (theta1, theta2) faces the camera (0..1).

    Dot product of the outward surface normal with the camera direction;
    used to dim trace segments on the far or inner side of the torus.
    """
    az, el = np.deg2rad(AZIM), np.deg2rad(ELEV)
    d = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
    nx, ny, nz = torus.normal(theta1, theta2)
    facing = nx * d[0] + ny * d[1] + nz * d[2]
    return 0.30 + 0.70 * (facing + 1.0) / 2.0


class DoublePendulumTorusFigure:
    """Builds the two-panel figure and exposes ``update(frame)`` for animation."""

    def __init__(self, traj: Trajectory, params: DoublePendulumParams,
                 dpi: int = 100, show_gravity: bool = True,
                 show_equilibrium: bool = True):
        self.traj = traj
        self.params = params
        self.show_gravity = show_gravity
        self.show_equilibrium = show_equilibrium
        self.n = len(traj.t)  # simulation samples (OVERSAMPLE per video frame)
        self.n_frames = int(np.ceil(self.n / OVERSAMPLE))

        p1x = params.l1 * np.sin(traj.theta1)
        p1y = -params.l1 * np.cos(traj.theta1)
        self.bob1_xy = np.column_stack((p1x, p1y))
        self.bob2_xy = np.column_stack((p1x + params.l2 * np.sin(traj.theta2),
                                        p1y - params.l2 * np.cos(traj.theta2)))
        tx, ty, tz = torus.embed(traj.theta1, traj.theta2)
        self.torus_pts = np.column_stack((tx, ty, tz))
        pts = self.torus_pts.reshape(-1, 1, 3)
        self.segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
        self.seg_facing = _facing_factor(
            0.5 * (traj.theta1[:-1] + traj.theta1[1:]),
            0.5 * (traj.theta2[:-1] + traj.theta2[1:]),
        )
        self.blue_rgb = np.array(to_rgb(BLUE))
        self.orange_rgb = np.array(to_rgb(ORANGE))

        self.fig = plt.figure(figsize=(12.8, 6.0), dpi=dpi)
        self.fig.patch.set_facecolor(SURFACE)
        gs = self.fig.add_gridspec(
            1, 2, width_ratios=[1.0, 1.25],
            left=0.03, right=0.99, top=0.99, bottom=0.03, wspace=0.02,
        )
        self.time_text = self.fig.text(
            0.02, 0.96, "", ha="left", va="center",
            fontsize=11, color=MUTED, family="monospace",
        )
        self._build_left(gs[0, 0])
        self._build_right(gs[0, 1])

    # ------------------------------------------------------------------ left
    def _build_left(self, spec) -> None:
        ax = self.fig.add_subplot(spec)
        self.ax_left = ax
        ax.set_facecolor(SURFACE)
        ax.set_aspect("equal")
        ax.set_xlim(-1.18, 1.18)
        ax.set_ylim(-1.24, 1.12)
        ax.axis("off")

        reach = self.params.l1 + self.params.l2
        phi = np.linspace(0, 2 * np.pi, 200)
        ax.plot(reach * np.cos(phi), reach * np.sin(phi), ls=(0, (3, 3)),
                lw=1.0, color=GUIDE, zorder=1)
        # theta = 0 reference direction (both links hanging down)
        ax.plot([0, 0], [0, -1.12], ls=(0, (1, 3)), lw=1.0, color=GUIDE, zorder=1)
        ax.text(0.04, -1.13, r"$\theta_1 = \theta_2 = 0$", fontsize=9,
                color=MUTED, va="top")
        # ceiling mount and pivot
        ax.plot([-0.22, 0.22], [0, 0], color=INK, lw=1.4, zorder=3)
        for xi in np.linspace(-0.19, 0.16, 6):
            ax.plot([xi, xi + 0.06], [0.0, 0.06], color=MUTED, lw=0.9, zorder=3)
        if self.show_gravity:
            ax.annotate("", xy=(1.02, -0.86), xytext=(1.02, -0.58),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
            ax.text(1.07, -0.72, r"$g$", fontsize=11, color=MUTED, va="center")

        self.trail = LineCollection([], linewidths=2.0, zorder=4)
        ax.add_collection(self.trail)
        (self.rod1,) = ax.plot([], [], color=INK, lw=2.2, zorder=5,
                               solid_capstyle="round")
        (self.rod2,) = ax.plot([], [], color=INK, lw=2.2, zorder=5,
                               solid_capstyle="round")
        ax.plot(0, 0, "o", ms=5, color=INK, zorder=6)
        (self.bob1,) = ax.plot([], [], "o", ms=10, color=ORANGE,
                               mec="white", mew=1.5, zorder=6)
        (self.bob2,) = ax.plot([], [], "o", ms=13, color=ORANGE,
                               mec="white", mew=1.5, zorder=7)
        self.state_text = ax.text(
            0.02, 0.02, "", transform=ax.transAxes, fontsize=10,
            color=MUTED, family="monospace",
        )

    # ----------------------------------------------------------------- right
    def _build_right(self, spec) -> None:
        ax = self.fig.add_subplot(spec, projection="3d", computed_zorder=False)
        self.ax_right = ax
        ax.set_facecolor(SURFACE)
        ax.view_init(elev=ELEV, azim=AZIM)
        ax.set_xlim(-1.55, 1.55)
        ax.set_ylim(-1.55, 1.55)
        ax.set_zlim(-0.62, 0.62)
        ax.set_box_aspect((1, 1, 0.40))
        ax.set_axis_off()

        # wireframe torus: meridians (tube cross-sections) and parallels
        for th1 in np.arange(-np.pi, np.pi, np.pi / 12):
            mx, my, mz = torus.meridian(th1)
            ax.plot(mx, my, mz, color=GUIDE, lw=0.6, alpha=0.45, zorder=1)
        for th2 in np.arange(-np.pi, np.pi, np.pi / 6):
            px, py, pz = torus.parallel_circle(th2)
            ax.plot(px, py, pz, color=GUIDE, lw=0.6, alpha=0.45, zorder=1)
        # reference circles: theta2 = 0 (outer equator) and theta1 = 0 meridian
        px, py, pz = torus.parallel_circle(0.0)
        ax.plot(px, py, pz, ls=(0, (4, 3)), color=MUTED, lw=0.9, alpha=0.55,
                zorder=2)
        mx, my, mz = torus.meridian(0.0)
        ax.plot(mx, my, mz, ls=(0, (4, 3)), color=MUTED, lw=0.9, alpha=0.55,
                zorder=2)
        ex, ey, ez = torus.embed(np.array([2.30]), np.array([0.0]))
        ax.text(ex[0] * 1.10, ey[0] * 1.10, ez[0] - 0.10, r"$\theta_2 = 0$",
                fontsize=8.5, color=MUTED, path_effects=HALO)
        mx, my, mz = torus.embed(np.array([0.0]), np.array([2.0]))
        ax.text(mx[0], my[0], mz[0] + 0.14, r"$\theta_1 = 0$",
                fontsize=8.5, color=MUTED, ha="center", path_effects=HALO)

        if self.show_equilibrium:
            # hanging equilibrium (0, 0): front of the outer equator
            ex, ey, ez = torus.embed(np.array([0.0]), np.array([0.0]))
            ax.plot([ex[0]], [ey[0]], [ez[0]], "o", ms=7, color=INK, zorder=6)
            ax.text(ex[0] - 0.42, ey[0], ez[0] - 0.14, "stable eq.", fontsize=9,
                    color=INK, ha="right", zorder=6, path_effects=HALO)

        seed = [self.segments[0] * 0.0 + self.torus_pts[0]]
        self.trace = Line3DCollection(seed, linewidths=1.9, zorder=5)
        ax.add_collection3d(self.trace)
        (self.dot3d,) = ax.plot([], [], [], "o", ms=9, color=ORANGE,
                                mec="white", mew=1.5, zorder=7)

    # ---------------------------------------------------------------- update
    def _trail_colors(self, k: int) -> np.ndarray:
        """Bob trail in orange, matching the bobs (the current state's color)."""
        alphas = np.linspace(0.0, 0.55, k)
        rgba = np.tile(np.append(self.orange_rgb, 0.0), (k, 1))
        rgba[:, 3] = alphas
        return rgba

    def _trace_colors(self, i: int, full_path: bool = False) -> np.ndarray:
        """RGBA colors for torus segments: depth-dimmed, age-faded."""
        facing = self.seg_facing[:i]
        if full_path:
            age = np.full(i, 0.85)
        else:
            age = np.full(i, 0.35)
            k = min(i, TRAIL_FRAMES * OVERSAMPLE)
            if k:
                age[-k:] = np.linspace(0.35, 0.95, k)
        rgba = np.tile(np.append(self.blue_rgb, 0.0), (i, 1))
        rgba[:, 3] = np.clip(age * facing, 0.0, 1.0)
        return rgba

    def _place_state_markers(self, i: int) -> None:
        b1, b2 = self.bob1_xy[i], self.bob2_xy[i]
        self.rod1.set_data([0, b1[0]], [0, b1[1]])
        self.rod2.set_data([b1[0], b2[0]], [b1[1], b2[1]])
        self.bob1.set_data([b1[0]], [b1[1]])
        self.bob2.set_data([b2[0]], [b2[1]])
        w1 = np.arctan2(np.sin(self.traj.theta1[i]), np.cos(self.traj.theta1[i]))
        w2 = np.arctan2(np.sin(self.traj.theta2[i]), np.cos(self.traj.theta2[i]))
        self.state_text.set_text(
            f"θ₁ = {w1:+5.2f} rad\nθ₂ = {w2:+5.2f} rad"
        )
        self.time_text.set_text(f"t = {self.traj.t[i]:5.2f} s")
        x, y, z = self.torus_pts[i]
        self.dot3d.set_data_3d([x], [y], [z])

    def update(self, frame: int):
        """Animation callback: state at video ``frame`` with an age-faded trace."""
        i = min(frame * OVERSAMPLE, self.n - 1)
        self._place_state_markers(i)

        k = min(i, TRAIL_FRAMES * OVERSAMPLE)
        if k:
            pts = self.bob2_xy[i - k: i + 1].reshape(-1, 1, 2)
            self.trail.set_segments(np.concatenate([pts[:-1], pts[1:]], axis=1))
            self.trail.set_color(self._trail_colors(k))
        else:
            self.trail.set_segments([])

        self.trace.set_segments(self.segments[:i])
        if i:
            self.trace.set_color(self._trace_colors(i))
        return ()

    def draw_summary(self, i_snap: int) -> None:
        """Full trace on the torus plus the pose at ``i_snap`` (for the PNG)."""
        self.trace.set_segments(self.segments)
        self.trace.set_color(self._trace_colors(len(self.segments), full_path=True))
        self._place_state_markers(i_snap)
        self.trail.set_segments([])
        x0, y0, z0 = self.torus_pts[0]
        self.ax_right.plot([x0], [y0], [z0], "o", ms=6, color=BLUE,
                           mec="white", mew=1.2, zorder=7)
        self.ax_right.text(x0, y0, z0 + 0.14, "start", fontsize=9, color=BLUE,
                           ha="center", zorder=7, path_effects=HALO)


# ------------------------------------------------------------------ rendering
def render_still(traj: Trajectory, params: DoublePendulumParams, path: Path,
                 t_snap: float, dpi: int = 200) -> None:
    """Summary image: full trace on the torus, pendulum pose at t = t_snap."""
    fig_obj = DoublePendulumTorusFigure(traj, params, dpi=dpi)
    i_snap = min(int(np.searchsorted(traj.t, t_snap)), fig_obj.n - 1)
    fig_obj.draw_summary(i_snap)
    fig_obj.fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig_obj.fig)


# ----------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render the double-pendulum-on-a-torus animation and figures."
    )
    parser.add_argument("--outdir", type=Path, default=Path("media"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--theta1-0", type=float, default=2.2)
    parser.add_argument("--theta2-0", type=float, default=-2.0)
    parser.add_argument("--theta1-dot0", type=float, default=0.0)
    parser.add_argument("--theta2-dot0", type=float, default=0.0)
    parser.add_argument("--damping1", type=float, default=0.0)
    parser.add_argument("--damping2", type=float, default=0.0)
    parser.add_argument("--t-snap", type=float, default=12.0,
                        help="time of the pendulum pose in the summary PNG")
    parser.add_argument("--no-video", action="store_true",
                        help="render only the summary PNG")
    args = parser.parse_args(argv)
    if args.duration * args.fps < 2:
        parser.error("duration must cover at least two frames (duration >= 2/fps)")

    args.outdir.mkdir(parents=True, exist_ok=True)
    params = DoublePendulumParams(damping1=args.damping1, damping2=args.damping2)
    traj = simulate(params, args.theta1_0, args.theta2_0,
                    args.theta1_dot0, args.theta2_dot0, args.duration,
                    args.fps * OVERSAMPLE)

    png = args.outdir / "double_pendulum_torus.png"
    print(f"Rendering summary image -> {png}")
    render_still(traj, params, png, t_snap=args.t_snap)

    if not args.no_video:
        mp4 = args.outdir / "double_pendulum_torus.mp4"
        gif = args.outdir / "double_pendulum_torus.gif"
        print(f"Rendering video -> {mp4}")
        fig_obj = DoublePendulumTorusFigure(traj, params)
        render_video(fig_obj.fig, fig_obj.update, fig_obj.n_frames, mp4,
                     fps=args.fps)
        plt.close(fig_obj.fig)
        print(f"Rendering GIF preview -> {gif}")
        mp4_to_gif(mp4, gif)
    print("Done.")


if __name__ == "__main__":
    main()
