"""Reproduction of Bullo, "Contraction Theory for Dynamical Systems",
Figure 3.7: phase portrait of the firing-rate model

    x' = f_FR(x, b) = -x + tanh(W x + b),
    W = [[0, 0.9], [-0.9, 0]],   b = [1, -1].

Jacobian: Df = -I + [d] W with [d] = diag(tanh'(Wx+b)), d in [0,1]^n.
Row-scaling by [d] gives, for the ell-inf log norm,

    mu_inf([d]W) = max_i d_i (w_ii + sum_{j!=i} |w_ij|) <= mu_inf(W),

so the one-sided Lipschitz constant obeys

    osLip(f) = sup_x mu_inf(Df(x)) <= -1 + mu_inf(W) = -1 + 0.9 = -0.1.

The system is strongly infinitesimally contracting w.r.t. ell-inf at
rate 0.1: there is a unique globally exponentially stable equilibrium
x* = tanh(W x* + b), and V(x) = ||x - x*||_inf is a global Lyapunov
function whose square level sets (dashed boxes) every streamline
crosses inward. Entry point: ``uv run firing-rate-portrait``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from ..style import BLUE, HALO, INK, MUTED, ORANGE, SURFACE

W = np.array([[0.0, 0.9], [-0.9, 0.0]])
B = np.array([1.0, -1.0])
RATE = 0.1                      # 1 - mu_inf(W)
BOX_RADII = [1.8, 1.0, 0.45]    # levels of V(x) = ||x - x*||_inf


def field(x1, x2):
    y1 = np.tanh(W[0, 0] * x1 + W[0, 1] * x2 + B[0])
    y2 = np.tanh(W[1, 0] * x1 + W[1, 1] * x2 + B[1])
    return -x1 + y1, -x2 + y2


def equilibrium(tol: float = 1e-14) -> np.ndarray:
    """x* = tanh(Wx* + b), by fixed-point iteration (a contraction)."""
    x = np.zeros(2)
    for _ in range(300):
        x_new = np.tanh(W @ x + B)
        if np.max(np.abs(x_new - x)) < tol:
            return x_new
        x = x_new
    return x


# ------------------------------------------------------------------ figure
def render_figure(xs: np.ndarray, path: Path, dpi: int = 200) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 8.6), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.90, bottom=0.135)
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal")
    for side in ax.spines.values():
        side.set_visible(False)

    lim = 2.55
    ax.set_xlim(xs[0] - lim, xs[0] + lim)
    ax.set_ylim(xs[1] - lim, xs[1] + lim)
    ax.axhline(0.0, color=INK, lw=1.0, zorder=2)
    ax.axvline(0.0, color=INK, lw=1.0, zorder=2)
    ax.tick_params(colors=MUTED, labelsize=9)

    g = np.linspace(xs[0] - lim, xs[0] + lim, 240)
    h = np.linspace(xs[1] - lim, xs[1] + lim, 240)
    X1, X2 = np.meshgrid(g, h)
    V1, V2 = field(X1, X2)
    ax.streamplot(X1, X2, V1, V2, color=BLUE, density=1.7, linewidth=0.8,
                  arrowsize=0.9, zorder=3)

    for r in BOX_RADII:                       # ell-inf level sets: squares
        ax.add_patch(Rectangle(xs - r, 2 * r, 2 * r, fc="none", ec=ORANGE,
                               lw=1.8, ls=(0, (5, 3)), zorder=5))
    ax.text(xs[0] + BOX_RADII[0] - 0.09, xs[1] + BOX_RADII[0] - 0.07,
            r"$\|x-x^*\|_\infty = c$", ha="right", va="top", fontsize=9.5,
            color=ORANGE, path_effects=HALO, zorder=6)

    ax.plot([xs[0]], [xs[1]], "o", ms=7, color=INK, zorder=6)
    ax.text(xs[0] + 0.12, xs[1] - 0.05, r"$x^*$", fontsize=13, color=INK,
            path_effects=HALO, zorder=6)
    ax.set_xlabel(r"$x_1$", fontsize=10, color=INK)
    ax.set_ylabel(r"$x_2$", fontsize=10, color=INK)

    ax.set_title(
        r"Firing-rate model $\dot x = -x + \tanh(Wx+b)$,  "
        r"$W=[\,0,\ 0.9\,;\ -0.9,\ 0\,]$,  $b=(1,-1)$",
        fontsize=12, color=INK, pad=12,
    )
    fig.text(0.5, 0.062,
             r"$\mu_\infty(W)=0.9<1 \Rightarrow \mathrm{osLip}(f) = "
             r"\sup_x \mu_\infty(-I+[d]W) \leq -1+\mu_\infty(W) = -0.1$: "
             r"strongly contracting in $\ell^\infty$.",
             ha="center", fontsize=9.5, color=MUTED)
    fig.text(0.5, 0.028,
             r"unique globally stable $x^*$; dashed squares: level sets of "
             r"$V=\|x-x^*\|_\infty$, crossed inward — "
             r"$V(t)\leq e^{-0.1t}V(0)$.",
             ha="center", fontsize=9.5, color=MUTED)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Phase portrait of the contracting firing-rate model "
                    "(Bullo Fig. 3.7)."
    )
    parser.add_argument("--outdir", type=Path, default=Path("media"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)

    xs = equilibrium()
    res = float(np.max(np.abs(xs - np.tanh(W @ xs + B))))
    print(f"x* = ({xs[0]:+.4f}, {xs[1]:+.4f}),  fixed-point residual {res:.1e}")

    # self-test: mu_inf(Df(x)) <= -0.1 on a dense state grid
    worst = -np.inf
    for x1 in np.linspace(xs[0] - 3, xs[0] + 3, 61):
        for x2 in np.linspace(xs[1] - 3, xs[1] + 3, 61):
            d = 1.0 / np.cosh(W @ np.array([x1, x2]) + B) ** 2
            J = -np.eye(2) + np.diag(d) @ W
            mu = np.max(np.diag(J) + np.sum(np.abs(J - np.diag(np.diag(J))),
                                            axis=1))
            worst = max(worst, mu)
    assert worst <= -RATE + 1e-12, f"certificate violated: {worst}"
    print(f"self-test: sup_x mu_inf(Df(x)) = {worst:.4f} <= -0.1  OK")

    args.outdir.mkdir(parents=True, exist_ok=True)
    png = args.outdir / "firing_rate_portrait.png"
    print(f"Rendering figure -> {png}")
    render_figure(xs, png, args.dpi)
    print("Done.")


if __name__ == "__main__":
    main()
