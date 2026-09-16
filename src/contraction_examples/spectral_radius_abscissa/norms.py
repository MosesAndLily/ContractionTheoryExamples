"""Norm/spectrum inequalities, after Figs. 2.2-2.3 of Bullo's
"Contraction Theory for Dynamical Systems" lecture notes.

Same five poles as ``plot.py``, drawn with the induced-2-norm and
log-norm bounds:

    top:     rho(A) = max|lambda|  <=  ||A||     (nested disks; the
             annulus between them is the "norm inefficiency")
    bottom:  -||A|| <= -mu(-A) <= min Re(lambda)
             <= alpha(A) = max Re(lambda) <= mu(A) <= ||A||
             (real-axis ticks; alpha-to-mu is the "logarithmic norm
             inefficiency")

The gaps depend on the realization, not just the spectrum: the
companion form of these poles has ||A|| ~ 100 (annulus off the page)
while the plain real modal form is normal (all gaps zero). So A is the
modal form plus a strictly upper-triangular coupling c*N — the spectrum
is untouched (block-triangular), and c tunes the non-normality to make
the gaps visible. Entry point: ``uv run norm-spectrum``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.linalg import block_diag

from ..style import BLUE, INK, MUTED, SURFACE
from .plot import POLES

ANNULUS = "#dcdbd6"  # light gray: rho <= |s| <= ||A||
DISK = "#aeada7"     # darker gray: |s| <= rho, holds the spectrum


def system_matrix(coupling: float) -> np.ndarray:
    """Real modal form + strictly upper-triangular coupling (spectrum
    unchanged; the coupling only adds non-normality)."""
    D = block_diag(POLES[0].real,
                   [[POLES[1].real, POLES[1].imag],
                    [-POLES[1].imag, POLES[1].real]],
                   POLES[3].real, POLES[4].real)
    N = np.triu(np.ones((5, 5)), k=1)
    N[1, 2] = 0.0  # inside the 2x2 modal block — keep it intact
    return D + coupling * N


DASHED = dict(arrowstyle="->", color=MUTED, lw=0.9, linestyle=(0, (4, 3)))


def _axis_cross(ax, x0: float, x1: float, y0: float, y1: float) -> None:
    arrow = dict(arrowstyle="-|>", color=MUTED, lw=1.6,
                 shrinkA=0.0, shrinkB=0.0, mutation_scale=16)
    ax.annotate("", xy=(x1, 0.0), xytext=(x0, 0.0), arrowprops=arrow,
                zorder=3, annotation_clip=False)
    ax.annotate("", xy=(0.0, y1), xytext=(0.0, y0), arrowprops=arrow,
                zorder=3, annotation_clip=False)


def render_figure(A: np.ndarray, path: Path, dpi: int = 200) -> None:
    rho = float(np.max(np.abs(POLES)))
    alpha = float(np.max(POLES.real))
    min_re = float(np.min(POLES.real))
    nrm = float(np.linalg.norm(A, 2))
    sym_ev = np.linalg.eigvalsh(0.5 * (A + A.T))
    mu, neg_mu_neg = float(sym_ev[-1]), float(sym_ev[0])  # mu(A), -mu(-A)

    fig = plt.figure(figsize=(7.6, 11.8), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[14.0, 7.8],
                          left=0.03, right=0.97, top=0.985, bottom=0.075,
                          hspace=0.05)

    # ------------------- top: rho(A) <= ||A|| as nested disks (Fig. 2.2)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal")
    ax.set_axis_off()
    lim = nrm + 1.4
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax.add_patch(Circle((0, 0), nrm, fc=ANNULUS, ec=INK, lw=1.0, zorder=1))
    ax.add_patch(Circle((0, 0), rho, fc=DISK, ec=INK, lw=1.0, zorder=2))
    _axis_cross(ax, -lim + 0.2, lim - 0.2, -lim + 0.2, lim - 0.2)
    ax.plot(POLES.real, POLES.imag, "x", ms=11, mew=2.6, color=BLUE,
            ls="none", zorder=5)
    for x in (rho, nrm):
        ax.plot([x], [0.0], marker="|", ms=11, mew=1.8, color=INK, zorder=4)

    ax.annotate(rf"$\rho(A)=\max|\lambda|={rho:g}$",
                xy=(rho, 0.30), xytext=(1.6, nrm + 0.55),
                ha="center", va="bottom", fontsize=11.5, color=INK,
                arrowprops=DASHED, zorder=6)
    ax.annotate(rf"$\|A\|={nrm:.2f}$",
                xy=(nrm, 0.30), xytext=(nrm + 0.2, nrm - 0.9),
                ha="center", va="bottom", fontsize=11.5, color=INK,
                arrowprops=DASHED, zorder=6, annotation_clip=False)
    ax.annotate("", xy=(rho, -0.5), xytext=(nrm, -0.5),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0),
                zorder=6)
    ax.annotate("norm\ninefficiency",
                xy=(0.5 * (rho + nrm), -0.65), xytext=(nrm + 0.3, -2.7),
                ha="center", va="top", fontsize=11.5, color=INK,
                arrowprops=DASHED, zorder=6)
    ax.text(-2.4, -1.5, r"$\lambda\in\mathrm{spec}(A)$", ha="center",
            fontsize=11.5, color=BLUE, zorder=6)

    # ------- bottom: log-norm / norm bounds on the real axis (Fig. 2.3)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(SURFACE)
    ax2.set_aspect("equal")
    ax2.set_axis_off()
    ax2.set_xlim(-lim - 0.3, lim + 0.3)
    ax2.set_ylim(-3.9, 3.9)

    _axis_cross(ax2, -lim, lim, -3.6, 3.6)
    ax2.plot(POLES.real, POLES.imag, "x", ms=11, mew=2.6, color=BLUE,
             ls="none", zorder=5)
    for x in (-nrm, neg_mu_neg, mu, nrm):
        ax2.plot([x], [0.0], marker="|", ms=11, mew=1.8, color=INK, zorder=4)

    ax2.text(-nrm, 0.45, r"$-\|A\|$", ha="center", va="bottom",
             fontsize=11.5, color=INK)
    ax2.text(neg_mu_neg, 1.35, r"$-\mu(-A)$", ha="center", va="bottom",
             fontsize=11.5, color=INK)
    ax2.text(min_re, -0.55, r"$\min\,\Re(\lambda)$", ha="center", va="top",
             fontsize=11.5, color=BLUE)
    ax2.annotate(r"$\alpha(A)=\max\,\Re(\lambda)$",
                 xy=(alpha, 0.30), xytext=(alpha + 1.4, 2.6),
                 ha="left", va="bottom", fontsize=11.5, color=BLUE,
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9,
                                 linestyle=(0, (4, 3))),
                 zorder=6)
    ax2.text(mu, 0.45, r"$\mu(A)$", ha="center", va="bottom",
             fontsize=11.5, color=INK)
    ax2.text(nrm, 0.45, r"$\|A\|$", ha="center", va="bottom",
             fontsize=11.5, color=INK)

    ax2.annotate("", xy=(alpha, -0.5), xytext=(mu, -0.5),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0),
                 zorder=6)
    ax2.text(0.5 * (alpha + mu) + 1.2, -1.0, "logarithmic norm\ninefficiency",
             ha="center", va="top", fontsize=11.5, color=INK)
    ax2.text(-1.8, 2.6, r"$\lambda\in\mathrm{spec}(A)$", ha="right",
             fontsize=11.5, color=BLUE)

    fig.text(0.5, 0.022,
             rf"$\rho(A)={rho:g}$, $\|A\|_2={nrm:.2f}$, "
             rf"$\alpha(A)=+{alpha:g}$, $\mu_2(A)={mu:+.2f}$, "
             rf"$-\mu_2(-A)={neg_mu_neg:+.2f}$, "
             rf"$\min\Re(\lambda)={min_re:g}$",
             ha="center", fontsize=9.5, color=MUTED)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Norm/spectrum inequalities (rho vs ||A||, alpha vs "
                    "mu) for the 5th-order example system."
    )
    parser.add_argument("--outdir", type=Path, default=Path("media"))
    parser.add_argument("--coupling", type=float, default=1.5,
                        help="strictly upper-triangular coupling strength; "
                             "0 makes A normal and every gap vanishes")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)

    A = system_matrix(args.coupling)
    # self-test: the coupling is block-triangular, so the spectrum is intact
    assert np.allclose(np.sort_complex(np.linalg.eigvals(A)),
                       np.sort_complex(POLES), atol=1e-9), \
        "coupling changed the spectrum"
    print("self-test: eig(A) == specified poles  OK")

    rho = float(np.max(np.abs(POLES)))
    alpha = float(np.max(POLES.real))
    nrm = float(np.linalg.norm(A, 2))
    sym_ev = np.linalg.eigvalsh(0.5 * (A + A.T))
    mu, neg_mu_neg = float(sym_ev[-1]), float(sym_ev[0])
    # self-test: the inequality chain the figure illustrates
    assert -nrm <= neg_mu_neg <= float(np.min(POLES.real)) \
        and alpha <= mu <= nrm, "norm/spectrum inequality chain violated"
    print(f"rho(A) = {rho:.3f} <= ||A|| = {nrm:.3f}")
    print(f"-||A|| = {-nrm:+.3f} <= -mu(-A) = {neg_mu_neg:+.3f} <= "
          f"min Re = {np.min(POLES.real):+.3f} <= alpha = {alpha:+.3f} <= "
          f"mu(A) = {mu:+.3f} <= ||A|| = {nrm:+.3f}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    png = args.outdir / "norm_spectrum_inequalities.png"
    print(f"Rendering figure -> {png}")
    render_figure(A, png, args.dpi)
    print("Done.")


if __name__ == "__main__":
    main()
