"""Two-panel figure: a swinging pendulum (left) and its state trajectory
(theta, theta_dot) living on the tangent-bundle cylinder TS^1 (right).

Renders an MP4 video, a GIF preview, and a summary PNG into an output
directory. Entry point: ``uv run pendulum-cylinder``.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib import patheffects
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from . import cylinder
from .dynamics import PendulumParams, Trajectory, simulate

# Validated categorical palette (dataviz reference instance, light mode).
BLUE = "#2a78d6"    # trajectory in state space
ORANGE = "#eb6834"  # the current state (bob + point on the cylinder)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GUIDE = "#c9c8c3"

VEL_SCALE = 7.0        # rad/s of theta_dot per unit of cylinder height
Z_MAX = 1.75           # cylinder half-height in embedding units
TRAIL_FRAMES = 45      # length of the bright "recent" window, in frames
ELEV, AZIM = 18.0, -78.0
# white casing so labels stay readable when the trajectory crosses them
HALO = [patheffects.withStroke(linewidth=2.5, foreground=SURFACE)]


def _facing_factor(theta: np.ndarray) -> np.ndarray:
    """How much the cylinder surface at angle theta faces the camera (0..1).

    The outward normal at angle theta is n = (sin t, -cos t); the camera's
    horizontal direction is d = (cos az, sin az). Segments on the far side
    get dimmed, since the translucent surface provides no occlusion cue.
    """
    az = np.deg2rad(AZIM)
    facing = np.sin(theta) * np.cos(az) - np.cos(theta) * np.sin(az)
    return 0.35 + 0.65 * (facing + 1.0) / 2.0


class PendulumCylinderFigure:
    """Builds the two-panel figure and exposes ``update(frame)`` for animation."""

    def __init__(self, traj: Trajectory, params: PendulumParams, dpi: int = 100):
        self.traj = traj
        self.params = params
        self.n = len(traj.t)

        L = params.length
        self.bob_xy = np.column_stack((L * np.sin(traj.theta), -L * np.cos(traj.theta)))
        cx, cy, cz = cylinder.embed(traj.theta, traj.theta_dot, VEL_SCALE)
        self.cyl_pts = np.column_stack((cx, cy, cz))
        pts = self.cyl_pts.reshape(-1, 1, 3)
        self.segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
        seg_theta = 0.5 * (traj.theta[:-1] + traj.theta[1:])
        self.seg_facing = _facing_factor(seg_theta)
        self.blue_rgb = np.array(to_rgb(BLUE))

        self.fig = plt.figure(figsize=(12.8, 6.0), dpi=dpi)
        self.fig.patch.set_facecolor(SURFACE)
        gs = self.fig.add_gridspec(
            1, 2, width_ratios=[1.0, 1.2],
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
        ax.set_xlim(-1.42, 1.42)
        ax.set_ylim(-1.52, 1.32)
        ax.axis("off")

        L = self.params.length
        # constraint circle of the bob and its label
        phi = np.linspace(0, 2 * np.pi, 200)
        ax.plot(L * np.cos(phi), L * np.sin(phi), ls=(0, (3, 3)), lw=1.0,
                color=GUIDE, zorder=1)
        ax.text(-0.76 * L, 0.80 * L, r"$S^1$", fontsize=11, color=MUTED,
                style="italic", ha="center")
        # theta = 0 reference direction (hanging down)
        ax.plot([0, 0], [0, -1.22 * L], ls=(0, (1, 3)), lw=1.0, color=GUIDE, zorder=1)
        ax.text(0.05, -1.24 * L, r"$\theta = 0$", fontsize=9, color=MUTED, va="top")
        # ceiling mount and pivot
        ax.plot([-0.30, 0.30], [0, 0], color=INK, lw=1.4, zorder=3)
        for xi in np.linspace(-0.26, 0.24, 7):
            ax.plot([xi, xi + 0.07], [0.0, 0.07], color=MUTED, lw=0.9, zorder=3)
        # gravity arrow
        ax.annotate("", xy=(1.24, -1.05), xytext=(1.24, -0.70),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
        ax.text(1.30, -0.88, r"$g$", fontsize=11, color=MUTED, va="center")

        self.trail = LineCollection([], linewidths=2.0, zorder=4)
        ax.add_collection(self.trail)
        (self.rod,) = ax.plot([], [], color=INK, lw=2.2, zorder=5,
                              solid_capstyle="round")
        ax.plot(0, 0, "o", ms=5, color=INK, zorder=6)
        (self.bob,) = ax.plot([], [], "o", ms=13, color=ORANGE,
                              mec="white", mew=1.5, zorder=7)
        self.vel_arrow = FancyArrowPatch(
            (0, 0), (0, 0), mutation_scale=13, color=ORANGE, lw=1.6,
            arrowstyle="-|>", zorder=6, alpha=0.9,
        )
        ax.add_patch(self.vel_arrow)
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
        ax.set_box_aspect((1, 1, 1.30))
        ax.set_xlim(-1.30, 1.30)
        ax.set_ylim(-1.30, 1.30)
        ax.set_zlim(-Z_MAX - 0.15, Z_MAX + 0.15)
        ax.set_axis_off()

        # translucent cylinder surface
        sx, sy, sz = cylinder.surface_mesh(-Z_MAX, Z_MAX)
        ax.plot_surface(sx, sy, sz, color="#b9b8b3", alpha=0.10,
                        linewidth=0, shade=False, zorder=1)
        # meridian guides and rims
        for th in np.arange(-np.pi, np.pi, np.pi / 4):
            f = _facing_factor(np.array([th]))[0]
            ax.plot([np.sin(th)] * 2, [-np.cos(th)] * 2, [-Z_MAX, Z_MAX],
                    color=GUIDE, lw=0.7, alpha=0.30 + 0.35 * f, zorder=2)
        for zc in (-Z_MAX, Z_MAX):
            rx, ry, rz = cylinder.rim_circle(zc)
            ax.plot(rx, ry, rz, color=GUIDE, lw=1.0, zorder=2)
        rx, ry, rz = cylinder.rim_circle(0.0)
        ax.plot(rx, ry, rz, ls=(0, (4, 3)), color=MUTED, lw=0.9, alpha=0.55, zorder=2)
        ax.text(np.sin(2.2) * 1.02, -np.cos(2.2) * 1.02, 0.10,
                r"$\dot{\theta} = 0$", fontsize=8.5, color=MUTED,
                path_effects=HALO)

        # angle labels on the bottom rim
        for th, lab in ((0.0, r"$\theta=0$"), (np.pi / 2, r"$\pi/2$"),
                        (np.pi, r"$\pm\pi$"), (-np.pi / 2, r"$-\pi/2$")):
            ax.text(1.22 * np.sin(th), -1.22 * np.cos(th), -Z_MAX - 0.22, lab,
                    fontsize=9, color=MUTED, ha="center")

        # velocity ruler at the cylinder's left silhouette edge, so it
        # projects outside the surface at this camera azimuth
        az = np.deg2rad(AZIM)
        edge = np.array([np.sin(az), -np.cos(az)])  # screen-left direction
        ax.plot([edge[0] * 1.45] * 2, [edge[1] * 1.45] * 2,
                [-10 / VEL_SCALE, 10 / VEL_SCALE], color=GUIDE, lw=1.0, zorder=2)
        for w in (-10, -5, 0, 5, 10):
            ax.plot([edge[0] * 1.41, edge[0] * 1.49],
                    [edge[1] * 1.41, edge[1] * 1.49],
                    [w / VEL_SCALE] * 2, color=GUIDE, lw=1.0, zorder=2)
            ax.text(*(edge * 1.64), w / VEL_SCALE, f"{w:+d}",
                    fontsize=8, color=MUTED, ha="center", va="center")
        ax.text(*(edge * 1.52), 10 / VEL_SCALE + 0.38,
                r"$\dot{\theta}$ [rad/s]", fontsize=9, color=MUTED, ha="center")

        # equilibria: stable (theta=0) faces the camera, unstable behind
        ax.plot([0], [-1], [0], "o", ms=7, color=INK, zorder=6)
        ax.text(-0.44, -1.02, -0.16, "stable eq.", fontsize=9, color=INK,
                ha="right", zorder=6, path_effects=HALO)
        ax.plot([0], [1], [0], "o", ms=7, mfc="none", mec=MUTED, zorder=2)
        ax.text(0, 1.06, 0.16, "unstable eq.", fontsize=8.5, color=MUTED,
                ha="center", alpha=0.6, zorder=2, path_effects=HALO)

        # seeded with a degenerate segment: add_collection3d cannot autoscale
        # from an empty collection
        seed = [self.segments[0] * 0.0 + self.cyl_pts[0]]
        self.traj3d = Line3DCollection(seed, linewidths=2.0, zorder=5)
        ax.add_collection3d(self.traj3d)
        (self.stem,) = ax.plot([], [], [], color=ORANGE, lw=1.4, alpha=0.55, zorder=6)
        (self.dot3d,) = ax.plot([], [], [], "o", ms=9, color=ORANGE,
                                mec="white", mew=1.5, zorder=7)

    # ---------------------------------------------------------------- update
    def _trail_colors(self, k: int) -> np.ndarray:
        """RGBA colors for the left-panel bob trail (recent window only)."""
        alphas = np.linspace(0.0, 0.55, k)
        rgba = np.tile(np.append(self.blue_rgb, 0.0), (k, 1))
        rgba[:, 3] = alphas
        return rgba

    def _traj_colors(self, i: int, full_path: bool = False) -> np.ndarray:
        """RGBA colors for cylinder segments: depth-dimmed, age-faded."""
        facing = self.seg_facing[:i]
        if full_path:
            age = np.full(i, 0.85)
        else:
            age = np.full(i, 0.40)
            k = min(i, TRAIL_FRAMES)
            if k:
                age[-k:] = np.linspace(0.40, 0.95, k)
        rgba = np.tile(np.append(self.blue_rgb, 0.0), (i, 1))
        rgba[:, 3] = np.clip(age * facing, 0.0, 1.0)
        return rgba

    def _place_state_markers(self, i: int) -> None:
        """Pendulum pose, velocity arrow, readouts, and cylinder point at index i."""
        th, w = self.traj.theta[i], self.traj.theta_dot[i]
        bx, by = self.bob_xy[i]
        self.rod.set_data([0, bx], [0, by])
        self.bob.set_data([bx], [by])
        tang = np.array([np.cos(th), np.sin(th)])
        tip = np.array([bx, by]) + 0.045 * w * tang
        self.vel_arrow.set_positions((bx, by), tuple(tip))
        self.vel_arrow.set_alpha(0.9 if abs(w) > 0.4 else 0.0)
        wrapped = np.arctan2(np.sin(th), np.cos(th))
        self.state_text.set_text(
            f"θ  = {wrapped:+5.2f} rad\nθ̇  = {w:+5.2f} rad/s"
        )
        self.time_text.set_text(f"t = {self.traj.t[i]:5.2f} s")
        x, y, z = self.cyl_pts[i]
        self.stem.set_data_3d([x, x], [y, y], [0.0, z])
        self.dot3d.set_data_3d([x], [y], [z])

    def update(self, frame: int):
        """Animation callback: state at ``frame`` with an age-faded trajectory."""
        i = min(frame, self.n - 1)
        self._place_state_markers(i)

        k = min(i, TRAIL_FRAMES)
        if k:
            pts = self.bob_xy[i - k: i + 1].reshape(-1, 1, 2)
            self.trail.set_segments(np.concatenate([pts[:-1], pts[1:]], axis=1))
            self.trail.set_color(self._trail_colors(k))
        else:
            self.trail.set_segments([])

        self.traj3d.set_segments(self.segments[:i])
        if i:
            self.traj3d.set_color(self._traj_colors(i))
        return ()

    def draw_summary(self, i_snap: int) -> None:
        """Full spiral on the cylinder plus the state at ``i_snap`` (for the PNG)."""
        self.traj3d.set_segments(self.segments)
        self.traj3d.set_color(self._traj_colors(len(self.segments), full_path=True))
        self._place_state_markers(i_snap)
        self.trail.set_segments([])
        x0, y0, z0 = self.cyl_pts[0]
        self.ax_right.plot([x0], [y0], [z0], "o", ms=6, color=BLUE,
                           mec="white", mew=1.2, zorder=7)
        self.ax_right.text(x0, y0, z0 + 0.22, "start", fontsize=9, color=BLUE,
                           ha="center", zorder=7)


# ------------------------------------------------------------------ rendering
def render_video(fig_obj: PendulumCylinderFigure, path: Path, fps: int,
                 hold_seconds: float = 1.2) -> None:
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    frames = fig_obj.n + int(hold_seconds * fps)

    def _progress(i, n):
        if i % 60 == 0 or i == n - 1:
            print(f"  frame {i + 1}/{n}")

    anim = animation.FuncAnimation(
        fig_obj.fig, fig_obj.update, frames=frames, interval=1000 / fps, blit=False
    )
    writer = animation.FFMpegWriter(
        fps=fps, codec="h264",
        extra_args=["-pix_fmt", "yuv420p", "-crf", "19", "-preset", "medium"],
    )
    anim.save(path, writer=writer, progress_callback=_progress)


def mp4_to_gif(mp4: Path, gif: Path, fps: int = 18, width: int = 880) -> None:
    """Small palette-optimized GIF preview generated from the MP4."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    vf = (
        f"fps={fps},scale={width}:-1:flags=lanczos,"
        "split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer"
    )
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(mp4), "-filter_complex", vf,
             "-loop", "0", str(gif)],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        print(exc.stderr.decode(errors="replace"))
        raise


def render_still(traj: Trajectory, params: PendulumParams, path: Path,
                 t_snap: float, dpi: int = 200) -> None:
    """Summary image: full trajectory on the cylinder, pendulum at t = t_snap."""
    fig_obj = PendulumCylinderFigure(traj, params, dpi=dpi)
    i_snap = min(int(np.searchsorted(traj.t, t_snap)), fig_obj.n - 1)
    fig_obj.draw_summary(i_snap)
    fig_obj.fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig_obj.fig)


# ----------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render the pendulum-on-a-cylinder animation and figures."
    )
    parser.add_argument("--outdir", type=Path, default=Path("media"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=14.0)
    parser.add_argument("--theta0", type=float, default=-np.pi / 2)
    parser.add_argument("--theta-dot0", type=float, default=11.0)
    parser.add_argument("--damping", type=float, default=0.55)
    parser.add_argument("--t-snap", type=float, default=3.57,
                        help="time of the pendulum pose in the summary PNG")
    parser.add_argument("--no-video", action="store_true",
                        help="render only the summary PNG")
    args = parser.parse_args(argv)
    if args.duration * args.fps < 2:
        parser.error("duration must cover at least two frames (duration >= 2/fps)")

    args.outdir.mkdir(parents=True, exist_ok=True)
    params = PendulumParams(damping=args.damping)
    traj = simulate(params, args.theta0, args.theta_dot0, args.duration, args.fps)

    png = args.outdir / "pendulum_cylinder.png"
    print(f"Rendering summary image -> {png}")
    render_still(traj, params, png, t_snap=args.t_snap)

    if not args.no_video:
        mp4 = args.outdir / "pendulum_cylinder.mp4"
        gif = args.outdir / "pendulum_cylinder.gif"
        print(f"Rendering video -> {mp4}")
        fig_obj = PendulumCylinderFigure(traj, params)
        render_video(fig_obj, mp4, fps=args.fps)
        plt.close(fig_obj.fig)
        print(f"Rendering GIF preview -> {gif}")
        mp4_to_gif(mp4, gif)
    print("Done.")


if __name__ == "__main__":
    main()
