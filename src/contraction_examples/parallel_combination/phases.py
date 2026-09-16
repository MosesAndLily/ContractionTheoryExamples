"""State-plane portraits for the parallel-combination example.

Top row — the shared-metric pair of ``plot.py``: x' = A1 x, x' = A2 x,
and the mixture x' = ((A1+A2)/2) x. The SAME ellipses (level sets of
V = x'Px, P = diag(4,1)) are drawn in all three panels: every
streamline of every panel crosses them inward, because contraction in
a fixed metric survives mixing (mu_P = -0.20, -0.28, -0.42).

Bottom row — the no-shared-metric pair: C1 = [[-1,10],[0,-1]] and
C2 = C1' are each Hurwitz (alpha = -1, all streamlines eventually reach
the origin), yet their average [[-1,5],[5,-1]] is a SADDLE
(eigenvalues +4, -6): the mixture diverges along the (1,1) direction.
No single family of level sets could have certified both endpoints —
that is exactly what "no common P" means.

Entry point: ``uv run parallel-combination-phases``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from ..style import BLUE, HALO, INK, MUTED, ORANGE, SURFACE
from .plot import A1, A2, C1, C2, alpha, mu_P

RED = "#e34948"          # categorical slot 8: the unstable direction
LIM = 1.3
ELLIPSE_LEVELS = [1.0, 0.36]   # V = x'Px levels drawn on the top row


def _portrait(ax, M: np.ndarray, ellipses: bool) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal")
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axhline(0.0, color=MUTED, lw=0.7, zorder=2)
    ax.axvline(0.0, color=MUTED, lw=0.7, zorder=2)

    g = np.linspace(-LIM, LIM, 160)
    X, Y = np.meshgrid(g, g)
    U = M[0, 0] * X + M[0, 1] * Y
    V = M[1, 0] * X + M[1, 1] * Y
    ax.streamplot(X, Y, U, V, color=BLUE, density=1.1, linewidth=0.7,
                  arrowsize=0.8, zorder=3)

    if ellipses:                       # x'Px = c, P = diag(4,1)
        for c in ELLIPSE_LEVELS:
            ax.add_patch(Ellipse((0, 0), np.sqrt(c), 2 * np.sqrt(c),
                                 fc="none", ec=ORANGE, lw=1.6,
                                 ls=(0, (5, 3)), zorder=5))
    ax.plot([0], [0], "o", ms=5, color=INK, zorder=6)


# ------------------------------------------------------------------ figure
def render_figure(path: Path, dpi: int = 200) -> None:
    mix_A = 0.5 * (A1 + A2)
    mix_C = 0.5 * (C1 + C2)
    rows = [
        [(A1, rf"$\dot x = A_1x$,  $\mu_P(A_1)={mu_P(A1):+.2f}$", True),
         (A2, rf"$\dot x = A_2x$,  $\mu_P(A_2)={mu_P(A2):+.2f}$", True),
         (mix_A, r"$\dot x = \frac{1}{2}(A_1{+}A_2)x$,  "
                 rf"$\mu_P={mu_P(mix_A):+.2f}$", True)],
        [(C1, rf"$\dot x = C_1x$,  $\alpha={alpha(C1):+.0f}$  (stable)",
          False),
         (C2, rf"$\dot x = C_2x$,  $\alpha={alpha(C2):+.0f}$  (stable)",
          False),
         (mix_C, r"$\dot x = \frac{1}{2}(C_1{+}C_2)x$,  "
                 rf"$\alpha={alpha(mix_C):+.0f}$  UNSTABLE", False)],
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12.4, 9.0), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.865, bottom=0.085,
                        wspace=0.10, hspace=0.22)

    for r, row in enumerate(rows):
        for c, (M, title, ell) in enumerate(row):
            ax = axes[r, c]
            _portrait(ax, M, ellipses=ell)
            ax.set_title(title, fontsize=10, color=INK, pad=8)

    # unstable eigendirection of the bottom-right saddle
    ax = axes[1, 2]
    s = LIM / np.sqrt(2.0)
    ax.plot([-s, s], [-s, s], color=RED, lw=1.6, ls=(0, (4, 3)), zorder=5)
    ax.text(0.72, 0.52, "unstable\ndirection", fontsize=8.5, color=RED,
            ha="left", va="top", path_effects=HALO, zorder=6)

    fig.text(0.017, 0.66, r"shared metric $P=\mathrm{diag}(4,1)$",
             rotation=90, ha="left", va="center", fontsize=10.5, color=MUTED)
    fig.text(0.017, 0.28, "no shared metric", rotation=90, ha="left",
             va="center", fontsize=10.5, color=MUTED)

    fig.suptitle(
        r"Parallel combination in the state plane — the same $P$-ellipses "
        r"certify $A_1$, $A_2$, and every mixture",
        fontsize=12.5, color=INK, y=0.965,
    )
    fig.text(0.5, 0.048,
             r"top: the dashed ellipses $x^{\top}Px=c$ are crossed inward "
             r"by all three flows — one metric, closed under mixing.   "
             r"bottom: $C_1, C_2$ each stable, no common ellipses exist,",
             ha="center", fontsize=9.5, color=MUTED)
    fig.text(0.5, 0.018,
             r"and the average is a saddle: trajectories escape along the "
             r"$(1,1)$ direction — mixing two stable systems is NOT safe "
             r"without a shared metric.",
             ha="center", fontsize=9.5, color=MUTED)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Phase portraits of A1, A2, C1, C2 and their mixtures."
    )
    parser.add_argument("--outdir", type=Path, default=Path("media"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)

    mix_A, mix_C = 0.5 * (A1 + A2), 0.5 * (C1 + C2)
    assert mu_P(mix_A) < 0.0 and alpha(mix_C) > 0.0
    print(f"mu_P: A1 {mu_P(A1):+.3f}, A2 {mu_P(A2):+.3f}, "
          f"mixture {mu_P(mix_A):+.3f}  (all < 0)")
    print(f"alpha: C1 {alpha(C1):+.1f}, C2 {alpha(C2):+.1f}, "
          f"mixture {alpha(mix_C):+.1f}  (mixture unstable)")

    args.outdir.mkdir(parents=True, exist_ok=True)
    png = args.outdir / "parallel_combination_phases.png"
    print(f"Rendering figure -> {png}")
    render_figure(png, args.dpi)
    print("Done.")


if __name__ == "__main__":
    main()
