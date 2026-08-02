"""Contraction of the damped pendulum on its configuration manifold S^1.

The pendulum theta_dd = -(g/l) sin(theta) - c theta_dot with g/l = 2,
c = 1 linearizes at the hanging equilibrium to exactly the mass-spring-
damper of the previous example: A = [[0, 1], [-2, -1]]. Its Lyapunov
metric P (PA + A^T P = -I) extends to a contraction metric for the
nonlinear pendulum on the region where

    P A(theta) + A(theta)^T P < 0,   A(theta) = [[0, 1], [-2 cos(theta), -1]],

which holds for cos(theta) > (11 - 2 sqrt(10)) / 9 ~ 0.52. A fan of
initial angles inside that region therefore contracts monotonically in
the P-distance (with Delta theta measured on S^1), while the Euclidean
distance between the same trajectories transiently grows.

Left: the fan of pendulums on S^1 collapsing onto one motion.
Right: max pairwise distance between trajectories, Euclidean vs P.

Entry point: ``uv run pendulum-contraction``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..mass_spring_damper.dynamics import make_A
from ..mass_spring_damper.metrics import lyapunov_P, quad_form
from ..media_utils import mp4_to_gif, render_video
from ..pendulum_cylinder.dynamics import PendulumParams, simulate
from ..style import BLUE, GUIDE, HALO, INK, MUTED, ORANGE, SURFACE

RED = "#e34948"    # categorical slot 8: the Euclidean metric (the one that fails)
OVERSAMPLE = 4     # simulation samples per video frame


def contraction_region(P: np.ndarray, k: float, c: float) -> float:
    """Largest angle bound within which P A(theta) + A(theta)^T P < 0.

    A(theta) = [[0, 1], [-k cos(theta), -c]]. Returns theta_max such that
    the P-metric contracts for all |theta| < theta_max.
    """
    thetas = np.linspace(0.0, np.pi, 2000)
    for th in thetas:
        Ath = np.array([[0.0, 1.0], [-k * np.cos(th), -c]])
        if np.linalg.eigvalsh(P @ Ath + Ath.T @ P).max() >= 0.0:
            return th
    return np.pi


def wrap(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


class PendulumContractionFigure:
    """Builds the two-panel figure and exposes ``update(frame)`` for animation."""

    def __init__(self, trajs: list, params: PendulumParams, P: np.ndarray,
                 dpi: int = 100):
        self.params = params
        self.P = P
        self.t = trajs[0].t
        self.n = len(self.t)
        self.n_frames = int(np.ceil(self.n / OVERSAMPLE))
        self.N = len(trajs)
        # states (N, n, 2) and bob positions on the unit circle
        self.Z = np.stack([np.column_stack((tr.theta, tr.theta_dot))
                           for tr in trajs])
        self.bobs = np.stack([np.column_stack((np.sin(tr.theta),
                                               -np.cos(tr.theta)))
                              for tr in trajs])
        self.d_eye = self._max_pairwise(np.eye(2))
        self.d_p = self._max_pairwise(P)

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

    def _max_pairwise(self, M: np.ndarray) -> np.ndarray:
        """Max over trajectory pairs of the M-distance, Delta-theta on S^1."""
        dists = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                dz = self.Z[i] - self.Z[j]
                dz[:, 0] = wrap(dz[:, 0])
                dists.append(np.sqrt(quad_form(dz, M)))
        return np.max(dists, axis=0)

    # ------------------------------------------------------------------ left
    def _build_left(self, spec) -> None:
        ax = self.fig.add_subplot(spec)
        self.ax_left = ax
        ax.set_facecolor(SURFACE)
        ax.set_aspect("equal")
        ax.set_xlim(-1.42, 1.42)
        ax.set_ylim(-1.74, 1.36)
        ax.axis("off")

        phi = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(phi), np.sin(phi), ls=(0, (3, 3)), lw=1.0,
                color=GUIDE, zorder=1)
        ax.text(-0.76, 0.80, r"$S^1$", fontsize=11, color=MUTED,
                style="italic", ha="center")
        ax.plot([0, 0], [0, -1.22], ls=(0, (1, 3)), lw=1.0, color=GUIDE,
                zorder=1)
        ax.text(0.05, -1.24, r"$\theta = 0$", fontsize=9, color=MUTED,
                va="top")
        ax.plot(0, 0, "o", ms=5, color=INK, zorder=6)
        ax.annotate("", xy=(1.24, -1.02), xytext=(1.24, -0.70),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
        ax.text(1.30, -0.86, r"$g$", fontsize=11, color=MUTED, va="center")

        # ghosts of the initial fan
        for b0 in self.bobs[:, 0]:
            ax.plot(b0[0], b0[1], "o", ms=7, mfc="none", mec=ORANGE,
                    mew=1.2, alpha=0.55, zorder=3)
        ax.text(*(self.bobs[-1, 0] * 1.16), "fan at t = 0", fontsize=9,
                color=ORANGE, ha="center", path_effects=HALO, zorder=3)

        self.rods = [ax.plot([], [], color=INK, lw=1.6, alpha=0.55,
                             zorder=4, solid_capstyle="round")[0]
                     for _ in range(self.N)]
        self.dots = [ax.plot([], [], "o", ms=10, color=ORANGE, mec="white",
                             mew=1.2, zorder=5)[0] for _ in range(self.N)]
        (self.arc,) = ax.plot([], [], lw=4.0, color=MUTED, alpha=0.45,
                              zorder=2, solid_capstyle="round")

        k = params_k = self.params.gravity / self.params.length
        c = self.params.damping
        th_max = contraction_region(self.P, params_k, c)
        p = self.P
        self.spread_text = ax.text(
            0.01, 0.015,
            "",  # first line is the live spread readout, set in update()
            transform=ax.transAxes, fontsize=8.5, color=MUTED,
            family="monospace", va="bottom", path_effects=HALO,
        )
        self.info_template = (
            "max spread = {spread:4.2f} rad\n"
            f"θ̈ = −{k:g} sin θ − {c:g} θ̇   (m=1, g/l={k:g}, c={c:g})\n"
            f"A(θ) = [ 0  1 ; −{k:g}cos θ  −{c:g} ],   "
            f"P = [ {p[0, 0]:.2f}  {p[0, 1]:.2f} ; {p[0, 1]:.2f}  {p[1, 1]:.2f} ]\n"
            f"PA(θ)+A(θ)ᵀP ≺ 0 for |θ| < {th_max:.2f} rad — fan stays inside"
        )

    # ----------------------------------------------------------------- right
    def _build_right(self, spec) -> None:
        ax = self.fig.add_subplot(spec)
        self.ax_right = ax
        ax.set_facecolor(SURFACE)
        ax.set_yscale("log")
        ax.set_xlim(0.0, self.t[-1])
        ax.set_ylim(5e-3, 8)
        ax.set_xlabel("t  [s]", fontsize=10, color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelsize=9)

        # asymptotic rate of the linearization: Re(eig) = -c/2
        rate = self.params.damping / 2.0
        ref = self.d_p[0] * np.exp(-rate * self.t)
        ax.plot(self.t, ref, color=GUIDE, lw=1.0, zorder=2)
        ax.text(self.t[-1] * 0.80, ref[int(0.80 * self.n)] * 0.30,
                rf"$\propto e^{{-{rate:g}\,t}}$", fontsize=9, color=MUTED)

        (self.line_eye,) = ax.plot(
            [], [], ls=(0, (5, 3)), lw=1.8, color=RED, zorder=3,
            label=r"$\max_{i,j}\ \|\Delta\mathbf{x}_{ij}\|$ — not monotone",
        )
        (self.line_p,) = ax.plot(
            [], [], lw=1.8, color=BLUE, zorder=4,
            label=r"$\max_{i,j}\ \sqrt{\Delta\mathbf{x}_{ij}^{\top} P\, \Delta\mathbf{x}_{ij}}$ — monotone: contraction",
        )
        ax.text(0.03, 0.04,
                r"$\Delta\mathbf{x}_{ij} = (\Delta\theta$ on $S^1,\ \Delta\dot{\theta})$",
                transform=ax.transAxes, fontsize=9, color=MUTED)
        ax.legend(loc="upper right", fontsize=8.5, frameon=False,
                  labelcolor=INK, handlelength=1.6)

    # ---------------------------------------------------------------- update
    def _place_state(self, i: int) -> None:
        for rod, dot, bobs in zip(self.rods, self.dots, self.bobs):
            rod.set_data([0, bobs[i, 0]], [0, bobs[i, 1]])
            dot.set_data([bobs[i, 0]], [bobs[i, 1]])
        th = self.Z[:, i, 0]
        arc = np.linspace(th.min(), th.max(), 60)
        self.arc.set_data(1.10 * np.sin(arc), -1.10 * np.cos(arc))
        spread = th.max() - th.min()
        self.spread_text.set_text(self.info_template.format(spread=spread))
        self.time_text.set_text(f"t = {self.t[i]:5.2f} s")

    def update(self, frame: int):
        """Animation callback: the fan and both spread curves at ``frame``."""
        i = min(frame * OVERSAMPLE, self.n - 1)
        self._place_state(i)
        self.line_eye.set_data(self.t[: i + 1], self.d_eye[: i + 1])
        self.line_p.set_data(self.t[: i + 1], self.d_p[: i + 1])
        return ()

    def draw_summary(self, i_snap: int) -> None:
        """Fan pose at ``i_snap`` with full distance curves (for the PNG)."""
        self._place_state(i_snap)
        self.line_eye.set_data(self.t, self.d_eye)
        self.line_p.set_data(self.t, self.d_p)


# ------------------------------------------------------------------ rendering
def render_still(trajs: list, params: PendulumParams, P: np.ndarray,
                 path: Path, t_snap: float, dpi: int = 200) -> None:
    fig_obj = PendulumContractionFigure(trajs, params, P, dpi=dpi)
    i_snap = min(int(np.searchsorted(fig_obj.t, t_snap)), fig_obj.n - 1)
    fig_obj.draw_summary(i_snap)
    fig_obj.fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig_obj.fig)


# ----------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render the pendulum contraction-on-S^1 animation and "
                    "figures."
    )
    parser.add_argument("--outdir", type=Path, default=Path("media"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--g-over-l", type=float, default=2.0,
                        help="pendulum stiffness g/l (the k of the "
                             "linearization)")
    parser.add_argument("--c", type=float, default=1.0,
                        help="damping constant")
    parser.add_argument("--spread", type=float, default=0.9,
                        help="initial fan of angles is [-spread, +spread]")
    parser.add_argument("--n", type=int, default=7,
                        help="number of pendulums in the fan")
    parser.add_argument("--t-snap", type=float, default=1.2,
                        help="time of the fan pose in the summary PNG")
    parser.add_argument("--no-video", action="store_true",
                        help="render only the summary PNG")
    args = parser.parse_args(argv)
    if args.duration * args.fps < 2:
        parser.error("duration must cover at least two frames (duration >= 2/fps)")

    params = PendulumParams(gravity=args.g_over_l, length=1.0, damping=args.c)
    P = lyapunov_P(make_A(args.g_over_l, args.c))
    th_max = contraction_region(P, args.g_over_l, args.c)
    if args.spread >= th_max:
        print(f"warning: fan spread {args.spread:.2f} rad exceeds the "
              f"P-contraction region |theta| < {th_max:.2f} rad — "
              "the P-distance may not decay monotonically")

    theta0s = np.linspace(-args.spread, args.spread, args.n)
    trajs = [simulate(params, th0, 0.0, args.duration,
                      args.fps * OVERSAMPLE) for th0 in theta0s]

    args.outdir.mkdir(parents=True, exist_ok=True)
    png = args.outdir / "pendulum_contraction_s1.png"
    print(f"Rendering summary image -> {png}")
    render_still(trajs, params, P, png, t_snap=args.t_snap)

    if not args.no_video:
        mp4 = args.outdir / "pendulum_contraction_s1.mp4"
        gif = args.outdir / "pendulum_contraction_s1.gif"
        print(f"Rendering video -> {mp4}")
        fig_obj = PendulumContractionFigure(trajs, params, P)
        render_video(fig_obj.fig, fig_obj.update, fig_obj.n_frames, mp4,
                     fps=args.fps)
        plt.close(fig_obj.fig)
        print(f"Rendering GIF preview -> {gif}")
        mp4_to_gif(mp4, gif)
    print("Done.")


if __name__ == "__main__":
    main()
