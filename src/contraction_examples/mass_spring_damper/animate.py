"""Two-panel figure: the phase plane of the mass-spring-damper (left) and
the two candidate contraction metrics along the trajectory (right).

Left: the spiral z(t) with the Euclidean level set ||z||^2 = c through the
current state (a circle — not monotone: it stalls at turning points and,
for the default k = 2, transiently grows) and the Lyapunov level set
z^T P z = c (a tilted ellipse — it shrinks at every instant).

Right: both quadratic forms on a log scale; ||z||^2 wobbles against the
exponential envelope, z^T P z decays monotonically.

Entry point: ``uv run mass-spring-damper``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..media_utils import mp4_to_gif, render_video
from ..style import BLUE, GUIDE, HALO, INK, MUTED, ORANGE, SURFACE
from .dynamics import Trajectory, make_A, simulate, turning_indices
from .metrics import level_set_points, lyapunov_P, quad_form

RED = "#e34948"    # categorical slot 8: the Euclidean metric (the one that fails)
OVERSAMPLE = 4     # simulation samples per video frame
IDENTITY = np.eye(2)


class MassSpringDamperFigure:
    """Builds the two-panel figure and exposes ``update(frame)`` for animation."""

    def __init__(self, traj: Trajectory, A: np.ndarray, dpi: int = 100):
        self.traj = traj
        self.A = A
        self.P = lyapunov_P(A)
        self.n = len(traj.t)
        self.n_frames = int(np.ceil(self.n / OVERSAMPLE))
        self.v_eye = quad_form(traj.z, IDENTITY)
        self.v_p = quad_form(traj.z, self.P)
        self.turns = turning_indices(traj)
        # decay rate of ||z||^2 set by the spectral abscissa of A
        self.decay_rate = -2.0 * np.max(np.real(np.linalg.eigvals(A)))

        self.fig = plt.figure(figsize=(12.8, 6.0), dpi=dpi)
        self.fig.patch.set_facecolor(SURFACE)
        gs = self.fig.add_gridspec(
            1, 2, width_ratios=[1.0, 1.15],
            left=0.045, right=0.985, top=0.93, bottom=0.11, wspace=0.16,
        )
        self.time_text = self.fig.text(
            0.02, 0.965, "", ha="left", va="center",
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
        lim = 3.50
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axis("off")

        # light axes through the origin
        ax.plot([-lim, lim], [0, 0], color=GUIDE, lw=0.8, zorder=1)
        ax.plot([0, 0], [-lim, lim], color=GUIDE, lw=0.8, zorder=1)
        ax.text(lim - 0.08, -0.22, r"$x$", fontsize=11, color=MUTED)
        ax.text(0.10, lim - 0.24, r"$\dot{x}$", fontsize=11, color=MUTED)
        ax.plot(0, 0, "o", ms=4, color=INK, zorder=6)

        ax.plot(self.traj.z[:, 0], self.traj.z[:, 1], color=MUTED, lw=0.9,
                alpha=0.45, zorder=2)
        (self.circle,) = ax.plot([], [], ls=(0, (5, 3)), lw=1.8, color=RED,
                                 zorder=3,
                                 label=r"$\|z\|^2 = c$ — circle, not monotone")
        (self.ellipse,) = ax.plot([], [], lw=1.8, color=BLUE, zorder=4,
                                  label=r"$z^{\top}\!P\,z = c$ — ellipse, always shrinks")
        (self.dot,) = ax.plot([], [], "o", ms=10, color=ORANGE,
                              mec="white", mew=1.5, zorder=7)
        ax.legend(loc="upper right", fontsize=8.5, frameon=False,
                  labelcolor=INK, handlelength=1.6)

        k, c = -self.A[1, 0], -self.A[1, 1]
        s1, s2 = np.linalg.eigvalsh(self.A + self.A.T)
        growth = ("‖z‖ transiently grows" if s2 > 1e-9
                  else "‖z‖ stalls at ẋ = 0")
        p = self.P
        ax.text(
            0.01, 0.015,
            f"ż = Az,  A = [ 0  1 ; −{k:g} −{c:g} ]   (m=1, k={k:g}, c={c:g})\n"
            f"eig(A+Aᵀ) = {{{s1:+.2f}, {s2:+.2f}}}  →  {growth}\n"
            f"P = [ {p[0, 0]:.2f}  {p[0, 1]:.2f} ; {p[0, 1]:.2f}  {p[1, 1]:.2f} ],"
            "   PA + AᵀP = −I",
            transform=ax.transAxes, fontsize=8.5, color=MUTED,
            family="monospace", va="bottom", path_effects=HALO,
        )

    # ----------------------------------------------------------------- right
    def _build_right(self, spec) -> None:
        ax = self.fig.add_subplot(spec)
        self.ax_right = ax
        ax.set_facecolor(SURFACE)
        ax.set_yscale("log")
        ax.set_xlim(0.0, self.traj.t[-1])
        ax.set_ylim(2e-5, 30)
        ax.set_xlabel("t  [s]", fontsize=10, color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelsize=9)

        # turning points: where the Euclidean curve goes momentarily flat
        for i in self.turns:
            if i == 0:  # the start is itself a turning point when xdot0 = 0
                continue
            ax.axvline(self.traj.t[i], color=GUIDE, lw=0.7, ls=(0, (1, 3)),
                       zorder=1)
        # pure-exponential reference with the true decay rate of ||z||^2
        rate = self.decay_rate
        ref = self.v_p[0] * np.exp(-rate * self.traj.t)
        ax.plot(self.traj.t, ref, color=GUIDE, lw=1.0, zorder=2)
        exp_str = "e^{-t}" if abs(rate - 1.0) < 1e-9 else f"e^{{-{rate:g}\\,t}}"
        ax.text(self.traj.t[-1] * 0.76, ref[int(0.76 * self.n)] * 0.12,
                rf"$\propto {exp_str}$", fontsize=9, color=MUTED)

        (self.line_eye,) = ax.plot([], [], ls=(0, (5, 3)), lw=1.8, color=RED,
                                   zorder=3,
                                   label=r"$\|z(t)\|^2$ — not monotone (flats, even growth)")
        (self.line_p,) = ax.plot([], [], lw=1.8, color=BLUE, zorder=4,
                                 label=r"$z(t)^{\top}\!P\,z(t)$ — monotone, exponential")
        ax.legend(loc="upper right", fontsize=8.5, frameon=False,
                  labelcolor=INK, handlelength=1.6)

    # ---------------------------------------------------------------- update
    def _place_state(self, i: int) -> None:
        c = level_set_points(IDENTITY, self.v_eye[i])
        e = level_set_points(self.P, self.v_p[i])
        self.circle.set_data(c[0], c[1])
        self.ellipse.set_data(e[0], e[1])
        self.dot.set_data([self.traj.z[i, 0]], [self.traj.z[i, 1]])
        self.time_text.set_text(f"t = {self.traj.t[i]:5.2f} s")

    def update(self, frame: int):
        """Animation callback: level sets through the state at video ``frame``."""
        i = min(frame * OVERSAMPLE, self.n - 1)
        self._place_state(i)
        self.line_eye.set_data(self.traj.t[: i + 1], self.v_eye[: i + 1])
        self.line_p.set_data(self.traj.t[: i + 1], self.v_p[: i + 1])
        return ()

    def draw_summary(self, i_snap: int) -> None:
        """Full norm curves; a nested cascade of P-ellipses (one per second)
        and the circle + ellipse through the state at ``i_snap``."""
        self._place_state(i_snap)
        self.line_eye.set_data(self.traj.t, self.v_eye)
        self.line_p.set_data(self.traj.t, self.v_p)
        for t_level in np.arange(1.0, 6.0):
            i = min(int(np.searchsorted(self.traj.t, t_level)), self.n - 1)
            e = level_set_points(self.P, self.v_p[i])
            self.ax_left.plot(e[0], e[1], lw=1.1, color=BLUE, alpha=0.35,
                              zorder=3)
        z0 = self.traj.z[0]
        if i_snap != 0:
            self.ax_left.plot([z0[0]], [z0[1]], "o", ms=6, color=ORANGE,
                              mec="white", mew=1.2, zorder=6)
        self.ax_left.annotate("start", (z0[0], z0[1]),
                              textcoords="offset points", xytext=(8, 6),
                              fontsize=9, color=ORANGE, path_effects=HALO)


# ------------------------------------------------------------------ rendering
def render_still(traj: Trajectory, A: np.ndarray, path: Path, t_snap: float,
                 dpi: int = 200) -> None:
    fig_obj = MassSpringDamperFigure(traj, A, dpi=dpi)
    i_snap = min(int(np.searchsorted(traj.t, t_snap)), fig_obj.n - 1)
    fig_obj.draw_summary(i_snap)
    fig_obj.fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig_obj.fig)


# ----------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render the mass-spring-damper contraction-metric "
                    "animation and figures."
    )
    parser.add_argument("--outdir", type=Path, default=Path("media"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--k", type=float, default=2.0, help="spring constant")
    parser.add_argument("--c", type=float, default=1.0, help="damping constant")
    parser.add_argument("--x0", type=float, default=2.0)
    parser.add_argument("--xdot0", type=float, default=0.0)
    parser.add_argument("--t-snap", type=float, default=0.0,
                        help="time of the snapshot in the summary PNG; the "
                             "default t=0 shows the spiral leaving tangent "
                             "to the circle while cutting into the ellipse")
    parser.add_argument("--no-video", action="store_true",
                        help="render only the summary PNG")
    args = parser.parse_args(argv)
    if args.duration * args.fps < 2:
        parser.error("duration must cover at least two frames (duration >= 2/fps)")

    A = make_A(args.k, args.c)
    traj = simulate(A, [args.x0, args.xdot0], args.duration,
                    args.fps * OVERSAMPLE)

    args.outdir.mkdir(parents=True, exist_ok=True)
    png = args.outdir / "mass_spring_damper.png"
    print(f"Rendering summary image -> {png}")
    render_still(traj, A, png, t_snap=args.t_snap)

    if not args.no_video:
        mp4 = args.outdir / "mass_spring_damper.mp4"
        gif = args.outdir / "mass_spring_damper.gif"
        print(f"Rendering video -> {mp4}")
        fig_obj = MassSpringDamperFigure(traj, A)
        render_video(fig_obj.fig, fig_obj.update, fig_obj.n_frames, mp4,
                     fps=args.fps)
        plt.close(fig_obj.fig)
        print(f"Rendering GIF preview -> {gif}")
        mp4_to_gif(mp4, gif)
    print("Done.")


if __name__ == "__main__":
    main()
