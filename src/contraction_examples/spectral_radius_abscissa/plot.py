"""Spectral radius vs spectral abscissa of a 5th-order LTI system.

Places five poles on the complex plane,

    s1    = +0.8         real, right half-plane
    s2,3  = -1 +/- 3i    complex-conjugate pair, left half-plane
    s4    = -2.5         real, strictly negative
    s5    = -4.0         real, strictly negative

and draws the two scalar summaries of the spectrum:

    spectral abscissa   alpha(A) = max Re s_i = +0.8   (vertical line)
    spectral radius     rho(A)   = max |s_i|  =  4.0   (circle)

alpha decides continuous-time stability of the flow xdot = A x: here
alpha > 0, so the flow is unstable and no constant metric P can certify
contraction, since inf_P mu_P(A) = alpha(A). rho decides discrete-time
stability of the map x+ = A x (needs rho < 1) and says nothing about the
flow — here it is attained by the MOST stable pole, s = -4. The two
summaries are set by different poles on purpose.

Entry point: ``uv run spectral-radius-abscissa``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import companion

from ..style import BLUE, INK, MUTED, ORANGE, SURFACE

POLES = np.array([0.8, -1.0 + 3.0j, -1.0 - 3.0j, -2.5, -4.0])


def system_matrix(poles: np.ndarray) -> np.ndarray:
    """Real companion realization of the monic polynomial with these roots."""
    coeffs = np.real(np.poly(poles))  # conjugate pair -> real coefficients
    return companion(coeffs)


# ------------------------------------------------------------------ figure
XLIM, YLIM = (-5.2, 4.4), (-4.8, 4.8)


def _pole_plane(ax, poles: np.ndarray) -> None:
    """Shared panel base: axis cross, ticks, and the five pole markers."""
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal")
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.axhline(0.0, color=MUTED, lw=0.9)
    ax.axvline(0.0, color=MUTED, lw=0.9)
    ax.set_xticks(range(-5, 5))
    ax.set_yticks(range(-4, 5))
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.plot(poles.real, poles.imag, "x", ms=11, mew=2.4, color=INK, ls="none")


def render_figure(poles: np.ndarray, path: Path, dpi: int = 200) -> None:
    alpha = float(np.max(poles.real))
    rho = float(np.max(np.abs(poles)))
    s_alpha = poles[np.argmax(poles.real)]
    s_rho = poles[np.argmax(np.abs(poles))]

    fig, (ax_r, ax_a) = plt.subplots(1, 2, figsize=(12.6, 6.6), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.97, bottom=0.07,
                        wspace=0.08)

    # ------------------------- left panel: spectral radius (circle |s| = rho)
    _pole_plane(ax_r, poles)
    th = np.linspace(0.0, 2.0 * np.pi, 400)
    ax_r.plot(rho * np.cos(th), rho * np.sin(th), color=BLUE, lw=1.8)
    ax_r.plot(s_rho.real, s_rho.imag, "o", ms=18, mfc="none", mec=BLUE,
              mew=1.6, ls="none")

    # ------------- right panel: spectral abscissa (line Re s = alpha), with
    # everything to its left shaded: the half-plane holding the spectrum
    _pole_plane(ax_a, poles)
    ax_a.set_yticklabels([])
    ax_a.tick_params(axis="y", length=0)
    ax_a.axvspan(XLIM[0], alpha, color=ORANGE, alpha=0.08, lw=0)
    ax_a.axvline(alpha, color=ORANGE, lw=1.8, ls=(0, (6, 3)))
    ax_a.plot(s_alpha.real, s_alpha.imag, "o", ms=18, mfc="none", mec=ORANGE,
              mew=1.6, ls="none")

    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Spectral radius vs spectral abscissa of a 5th-order "
                    "LTI system with one RHP pole, one LHP complex pair, "
                    "and two strictly negative real poles."
    )
    parser.add_argument("--outdir", type=Path, default=Path("media"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)

    A = system_matrix(POLES)
    # self-test: the companion realization reproduces the specified poles
    assert np.allclose(np.sort_complex(np.linalg.eigvals(A)),
                       np.sort_complex(POLES), atol=1e-9), \
        "companion eigenvalues disagree with the specified poles"
    print("self-test: eig(A) == specified poles  OK")

    with np.printoptions(precision=3, suppress=True):
        print("companion A =\n", A)
    print(f"spectral abscissa alpha(A) = {np.max(POLES.real):+.3f} "
          "(attained by the RHP pole)")
    print(f"spectral radius   rho(A)   = {np.max(np.abs(POLES)):.3f} "
          "(attained by the most stable pole)")

    args.outdir.mkdir(parents=True, exist_ok=True)
    png = args.outdir / "spectral_radius_abscissa.png"
    print(f"Rendering figure -> {png}")
    render_figure(POLES, png, args.dpi)
    print("Done.")


if __name__ == "__main__":
    main()
