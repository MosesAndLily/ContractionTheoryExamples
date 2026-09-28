"""Superposition of three potentials for a first-order planar two-rod arm.

One column per potential — joint space U_q, task position U_x, task
orientation U_phi — and a fourth for their sum U = U_q + U_x + U_phi, each
driving the same fan of nine arms by b q_dot = -grad U(q).

Top: the fan rendered by MuJoCo (ghost arm = q*, blue dot/arrow = x*, phi*).
Middle: the configuration torus painted with the contraction rate
lambda_min(Hess U)/b (blue: contracting, red: expanding), the minima of U,
and the fan's trajectories. Bottom: max pairwise flat-torus distance.

Entry point: ``uv run rod-potentials``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from contourpy import contour_generator
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from ..double_pendulum_torus import torus
from ..double_pendulum_torus.animate import AZIM, ELEV, _facing_factor
from ..media_utils import mp4_to_gif, render_video
from ..style import BLUE, GUIDE, HALO, INK, MUTED, ORANGE, SURFACE
from .model import RodArms
from .potentials import (TERMS, RodPotentials, max_pairwise_distance,
                         segment_certificate, wrap)

RED = "#e34948"
OVERSAMPLE = 2       # simulation samples per video frame
LAM_CAP = 4.0        # symmetric clip of the painted rate field
DIVERGING = LinearSegmentedColormap.from_list(
    "contraction_div", [RED, "#f0efec", BLUE]
)
COLUMNS = (
    (("joint",), "joint space",
     r"$U_q = k_q\sum_i\,(1-\cos(q_i-q_i^*))$"),
    (("position",), "task position",
     r"$U_x = \frac{1}{2}k_x\,\|x(q)-x^*\|^2$"),
    (("orientation",), "task orientation",
     r"$U_\phi = k_\phi\,(1-\cos(q_1+q_2-\phi^*))$"),
    (TERMS, "superposition",
     r"$U = U_q + U_x + U_\phi$"),
)


class RodPotentialsFigure:
    """Builds the 3x4 figure and exposes ``update(frame)`` for animation."""

    def __init__(self, pot: RodPotentials, arms: RodArms, t: np.ndarray,
                 runs: list[np.ndarray], certs: list[float],
                 fields: list[np.ndarray], grid: tuple, dpi: int = 100,
                 render_px: int = 420):
        self.pot, self.arms, self.t, self.runs = pot, arms, t, runs
        self.n = len(t)
        self.n_frames = int(np.ceil(self.n / OVERSAMPLE))
        self.N = runs[0].shape[0]
        # torus chart centered between the two inverse-kinematics branches,
        # so q* and its elbow-flipped mirror both sit on the front
        self.center = np.array([0.5 * (pot.q_star[0] + pot.q_mirror[0]), 0.0])
        self.dists = [max_pairwise_distance(Q) for Q in runs]
        self.renderer = mujoco.Renderer(arms.model, render_px, render_px)
        self.orange_rgb = np.array(to_rgb(ORANGE))

        self.fig = plt.figure(figsize=(16.0, 11.2), dpi=dpi)
        self.fig.patch.set_facecolor(SURFACE)
        gs = self.fig.add_gridspec(
            3, 4, height_ratios=[1.0, 1.0, 0.72], left=0.05, right=0.99,
            top=0.885, bottom=0.11, wspace=0.12, hspace=0.16,
        )
        self.fig.text(
            0.5, 0.975,
            r"First-order flow  $b\,\dot{q} = -\nabla U(q)$  of a planar "
            r"two-rod arm on the torus $T^2$ — three potentials and their "
            "superposition",
            ha="center", va="center", fontsize=14, color=INK,
        )
        self.fig.text(
            0.5, 0.948,
            f"l₁ = l₂ = {pot.l1:g},  k_q = {pot.k_q:g},  k_x = {pot.k_x:g},  "
            f"k_φ = {pot.k_phi:g},  b = {pot.b:g},  "
            f"q* = ({pot.q_star[0]:g}, {pot.q_star[1]:g})   ·   MuJoCo: "
            "kinematics, site Jacobians Jₓ, J_φ and rendering; "
            "∇Uₓ = Jₓᵀkₓ(x − x*),  ∇U_φ = J_φᵀk_φ sin(φ − φ*)",
            ha="center", va="center", fontsize=9.5, color=MUTED,
            family="monospace",
        )
        self.time_text = self.fig.text(
            0.012, 0.975, "", ha="left", va="center", fontsize=11,
            color=MUTED, family="monospace",
        )

        self.images, self.traces, self.dots, self.pts3d = [], [], [], []
        self.facing, self.curves = [], []
        norm = TwoSlopeNorm(vmin=-LAM_CAP, vcenter=0.0, vmax=LAM_CAP)
        for c, (terms, name, formula) in enumerate(COLUMNS):
            self._build_render(gs[0, c], name, formula)
            self._build_torus(gs[1, c], c, terms, fields[c], grid, norm)
            self._build_distance(gs[2, c], c, terms, certs[c])

        cax = self.fig.add_axes([0.30, 0.045, 0.40, 0.013])
        cbar = self.fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=DIVERGING), cax=cax,
            orientation="horizontal", extend="both",
        )
        cbar.set_label(
            r"contraction rate $\lambda_{\min}(\nabla^2 U)/b$ on $T^2$ "
            r"(flat metric; $>0$: contracting)   —   Weyl: "
            r"$\lambda_{\min}(\sum_i \nabla^2 U_i) \geq "
            r"\sum_i \lambda_{\min}(\nabla^2 U_i)$",
            fontsize=9, color=INK,
        )
        cbar.ax.tick_params(colors=MUTED, labelsize=8)
        cbar.outline.set_edgecolor(MUTED)
        cbar.outline.set_linewidth(0.8)

    # --------------------------------------------------------------- helpers
    def _embed(self, q1, q2, lift: float = 0.0):
        t1, t2 = q1 - self.center[0], q2 - self.center[1]
        x, y, z = torus.embed(t1, t2)
        if lift:
            nx, ny, nz = torus.normal(t1, t2)
            return x + lift * nx, y + lift * ny, z + lift * nz
        return x, y, z

    # ------------------------------------------------------------------- top
    def _build_render(self, spec, name: str, formula: str) -> None:
        ax = self.fig.add_subplot(spec)
        ax.set_axis_off()
        ax.set_title(f"{name}\n{formula}", fontsize=11.5, color=INK, pad=4)
        blank = np.full((self.renderer.height, self.renderer.width, 3), 252,
                        dtype=np.uint8)
        self.images.append(ax.imshow(blank, interpolation="lanczos"))

    # ---------------------------------------------------------------- middle
    def _build_torus(self, spec, c: int, terms, field, grid, norm) -> None:
        ax = self.fig.add_subplot(spec, projection="3d", computed_zorder=False)
        ax.set_facecolor(SURFACE)
        ax.view_init(elev=ELEV, azim=AZIM)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-0.5, 0.5)
        ax.set_box_aspect((1, 1, 0.40))
        ax.set_axis_off()
        pot = self.pot

        Q1, Q2 = grid
        sx, sy, sz = self._embed(Q1, Q2)
        colors = DIVERGING(norm(np.clip(field, -LAM_CAP, LAM_CAP)))
        ax.plot_surface(sx, sy, sz, facecolors=colors, rstride=1, cstride=1,
                        shade=False, antialiased=False, linewidth=0, zorder=1)
        if field.max() > 1e-6 and field.min() < -1e-6:
            for line in contour_generator(x=Q1, y=Q2, z=field).lines(0.0):
                x, y, z = self._embed(line[:, 0], line[:, 1], lift=0.012)
                ax.plot(x, y, z, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=2)

        def mark(q, label, dx=0.0, dz=-0.2, hollow=False):
            x, y, z = self._embed(np.array([q[0]]), np.array([q[1]]), 0.03)
            ax.plot(x, y, z, "o", ms=7, zorder=8,
                    **({"mfc": SURFACE, "mec": INK, "mew": 1.4} if hollow
                       else {"color": INK}))
            ax.text(x[0] + dx, y[0], z[0] + dz, label, fontsize=8.5,
                    color=INK, ha="center", zorder=9, path_effects=HALO)

        if terms == ("orientation",):
            # whole circle of minima: q1 + q2 = phi*, a (1,1) curve on T^2
            s = np.linspace(-np.pi, np.pi, 240)
            x, y, z = self._embed(s, pot.phi_star - s, lift=0.02)
            ax.plot(x, y, z, color=INK, lw=1.6, zorder=6)
            ax.text2D(0.5, 0.10, r"minima: the whole circle $q_1+q_2=\phi^*$",
                      transform=ax.transAxes, fontsize=8.5, color=INK,
                      ha="center")
        else:
            mark(pot.q_star, r"$q^*$")
            if terms == ("position",):
                mark(pot.q_mirror, "elbow-flipped\nmirror", dz=-0.36,
                     hollow=True)

        for k in range(self.N):
            Q = self.runs[c][k]
            x, y, z = self._embed(Q[:, 0], Q[:, 1], lift=0.018)
            self.pts3d.append(np.column_stack((x, y, z)))
            t1 = Q[:, 0] - self.center[0]
            t2 = Q[:, 1] - self.center[1]
            self.facing.append(_facing_factor(0.5 * (t1[:-1] + t1[1:]),
                                              0.5 * (t2[:-1] + t2[1:])))
            p0 = self.pts3d[-1][0]
            ax.plot([p0[0]], [p0[1]], [p0[2]], "o", ms=5, mfc="none",
                    mec=ORANGE, mew=1.1, alpha=0.75, zorder=3)
            lc = Line3DCollection([np.stack([p0, p0])], linewidths=1.4,
                                  zorder=5)
            ax.add_collection3d(lc)
            self.traces.append(lc)
            self.dots.append(ax.plot([], [], [], "o", ms=6, color=ORANGE,
                                     mec="white", mew=1.0, zorder=7)[0])

    # ---------------------------------------------------------------- bottom
    def _build_distance(self, spec, c: int, terms, cert: float) -> None:
        ax = self.fig.add_subplot(spec)
        ax.set_facecolor(SURFACE)
        ax.set_yscale("log")
        ax.set_xlim(0.0, self.t[-1])
        d = self.dists[c]
        ax.set_ylim(1e-6, 10.0)
        ax.set_xlabel("t  [s]", fontsize=9.5, color=INK)
        if c == 0:
            ax.set_ylabel(r"$\max_{i,j}\ \|q_i - q_j\|_{T^2}$", fontsize=10,
                          color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelsize=8.5)

        # local rate at q*: slowest eigenvalue of Hess U(q*)/b
        lam_star = float(self.pot.lambda_min(terms, np.asarray(self.pot.q_star)))
        ref = d[0] * np.exp(-lam_star * self.t)
        ax.plot(self.t, ref, color=GUIDE, lw=1.0, zorder=2)
        if cert > 1e-9:
            ax.plot(self.t, d[0] * np.exp(-cert * self.t), color=MUTED,
                    lw=1.0, ls=(0, (6, 3)), alpha=0.7, zorder=2)
            verdict = (f"certified: λ_min ≥ {cert:.2f} on all segments\n"
                       f"dashed: guaranteed e^(−{cert:.2f} t)")
        elif cert > -1e-9:
            verdict = ("semi-contracting only: λ_min = 0\n"
                       "(flat along the circle of minima)")
        else:
            verdict = (f"not certified: λ_min reaches {cert:+.2f}\n"
                       "on inter-trajectory segments")
        ax.text(0.03, 0.04,
                f"{verdict}\ngrey: local rate at q*, e^(−{lam_star:.2f} t)",
                transform=ax.transAxes, fontsize=8, color=MUTED, ha="left",
                va="bottom", family="monospace", path_effects=HALO)
        (line,) = ax.plot([], [], lw=1.8, color=BLUE, zorder=4)
        self.curves.append(line)

    # ---------------------------------------------------------------- update
    def _draw(self, i_pose: int, i_trace: int) -> None:
        for c, Q in enumerate(self.runs):
            self.arms.pose_for_render(Q[:, i_pose])
            self.renderer.update_scene(self.arms.data, camera="top")
            self.images[c].set_data(self.renderer.render())
            self.curves[c].set_data(self.t[: i_trace + 1],
                                    self.dists[c][: i_trace + 1])
        for k, (pts, lc) in enumerate(zip(self.pts3d, self.traces)):
            x, y, z = pts[i_pose]
            self.dots[k].set_data_3d([x], [y], [z])
            seg_pts = pts[: i_trace + 1].reshape(-1, 1, 3)
            if len(seg_pts) < 2:
                lc.set_segments([])
                continue
            segs = np.concatenate([seg_pts[:-1], seg_pts[1:]], axis=1)
            rgba = np.tile(np.append(self.orange_rgb, 0.0), (len(segs), 1))
            rgba[:, 3] = 0.85 * self.facing[k][: len(segs)]
            lc.set_segments(segs)
            lc.set_color(rgba)
        self.time_text.set_text(f"t = {self.t[i_pose]:5.2f} s")

    def update(self, frame: int):
        """Animation callback: every column at video ``frame``."""
        i = min(frame * OVERSAMPLE, self.n - 1)
        self._draw(i, i)
        return ()

    def draw_summary(self, i_snap: int) -> None:
        """Full traces and curves with the fan pose at ``i_snap``."""
        self._draw(i_snap, self.n - 1)
        self.time_text.set_text("")

    def close(self) -> None:
        self.renderer.close()
        plt.close(self.fig)


# ----------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render the three-potential superposition example for "
                    "a first-order planar two-rod arm (MuJoCo)."
    )
    parser.add_argument("--outdir", type=Path, default=Path("media"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--k-q", type=float, default=1.0)
    parser.add_argument("--k-x", type=float, default=2.0)
    parser.add_argument("--k-phi", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=1.0,
                        help="first-order damping (b q_dot = -grad U)")
    parser.add_argument("--q-star", type=float, nargs=2, default=(0.4, 1.2),
                        metavar=("Q1", "Q2"),
                        help="goal configuration; x* and phi* are its tip "
                             "position and orientation")
    parser.add_argument("--spread", type=float, default=0.6,
                        help="fan is the 3x3 grid q* + [-spread, spread]^2; "
                             "try 1.2 to see U_x alone send an arm to the "
                             "elbow-flipped mirror")
    parser.add_argument("--t-snap", type=float, default=0.6,
                        help="time of the fan pose in the summary PNG")
    parser.add_argument("--no-video", action="store_true",
                        help="render only the summary PNG")
    args = parser.parse_args(argv)
    if args.duration * args.fps < 2:
        parser.error("duration must cover at least two frames (duration >= 2/fps)")

    pot = RodPotentials(k_q=args.k_q, k_x=args.k_x, k_phi=args.k_phi,
                        b=args.b, q_star=tuple(args.q_star))
    s = args.spread
    q0 = np.asarray(pot.q_star) + np.array(
        [(a, b) for a in (-s, 0.0, s) for b in (-s, 0.0, s)])
    arms = RodArms(pot, len(q0))
    err = arms.self_test(np.random.default_rng(0))
    print(f"MuJoCo vs closed-form gradients: max |difference| = {err:.1e}")

    n_samples = int(round(args.duration * args.fps * OVERSAMPLE))
    runs, certs = [], []
    for terms, name, _ in COLUMNS:
        t, Q = arms.simulate(terms, q0, args.duration, n_samples)
        cert = segment_certificate(pot, terms, Q)
        runs.append(Q)
        certs.append(cert)
        ends = np.unique(np.round(wrap(Q[:, -1]), 1), axis=0)
        print(f"  {name:17s} segment certificate {cert:+.3f}   "
              f"final distance {max_pairwise_distance(Q)[-1]:.1e}   "
              f"{len(ends)} distinct end configuration(s)")

    g = np.linspace(-np.pi, np.pi, 73)
    center1 = 0.5 * (pot.q_star[0] + pot.q_mirror[0])
    Q1, Q2 = np.meshgrid(g + center1, g, indexing="ij")
    qgrid = np.stack([Q1, Q2], -1)
    fields = [pot.lambda_min(terms, qgrid) for terms, _, _ in COLUMNS]

    args.outdir.mkdir(parents=True, exist_ok=True)
    png = args.outdir / "rod_potentials.png"
    print(f"Rendering summary image -> {png}")
    fig_obj = RodPotentialsFigure(pot, arms, t, runs, certs, fields, (Q1, Q2),
                                  dpi=150, render_px=640)
    i_snap = min(int(np.searchsorted(t, args.t_snap)), len(t) - 1)
    fig_obj.draw_summary(i_snap)
    fig_obj.fig.savefig(png, dpi=150, facecolor=SURFACE)
    fig_obj.close()

    if not args.no_video:
        mp4 = args.outdir / "rod_potentials.mp4"
        gif = args.outdir / "rod_potentials.gif"
        print(f"Rendering video -> {mp4}")
        fig_obj = RodPotentialsFigure(pot, arms, t, runs, certs, fields,
                                      (Q1, Q2))
        render_video(fig_obj.fig, fig_obj.update, fig_obj.n_frames, mp4,
                     fps=args.fps)
        fig_obj.close()
        print(f"Rendering GIF preview -> {gif}")
        mp4_to_gif(mp4, gif, width=1000)
    print("Done.")


if __name__ == "__main__":
    main()
