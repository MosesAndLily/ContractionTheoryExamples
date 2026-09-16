"""Unit-ball picture of induced norms ||A||_p and log norms mu_p(A).

Top row (the classic induced-norm picture): map the unit ball of each
norm through A; ||A||_p is the smallest dilation of the ball that
contains the image — the maximal stretch.

Bottom row (the log-norm analogue): flow the unit ball under xdot = Ax
for a short time h. The containment

    e^{hA} B_p  <=  e^{mu_p(A) h} B_p      for ALL h >= 0

makes mu_p(A) the initial exponential growth rate of the smallest
containing dilation: mu_p = lim_{h->0+} (||I + hA||_p - 1)/h.

Example matrix: a damped rotation A = [[-0.2, 2], [-2, -0.2]], with
alpha(A) = -0.2 (asymptotically stable). It is normal, so mu_2 = alpha
= -0.2 < 0: the ell^2 ball strictly shrinks. But the rotation initially
inflates the ell^1 diamond and ell^inf square: mu_1 = mu_inf = +1.8 > 0.
Same system, three different instantaneous verdicts — the log norm is a
property of the pair (system, norm). Closed forms:

    mu_1(A)   = max_j ( a_jj + sum_{i != j} |a_ij| )   (column rule)
    mu_2(A)   = lambda_max( (A + A^T)/2 )
    mu_inf(A) = max_i ( a_ii + sum_{j != i} |a_ij| )   (row rule)

Entry point: ``uv run induced-log-norms``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from ..style import BLUE, INK, MUTED, ORANGE, SURFACE

A_DEFAULT = np.array([[-0.2, 2.0], [-2.0, -0.2]])

_TH = np.linspace(0.0, 2.0 * np.pi, 721)
_CIRCLE = np.vstack([np.cos(_TH), np.sin(_TH)])


def ball_boundary(p) -> np.ndarray:
    """Boundary of the unit p-norm ball (radially normalized circle)."""
    return _CIRCLE / np.linalg.norm(_CIRCLE, ord=p, axis=0)


def log_norm(A: np.ndarray, p) -> float:
    off = np.abs(A) - np.diag(np.abs(np.diag(A)))
    if p == 1:
        return float(np.max(np.diag(A) + off.sum(axis=0)))
    if p == 2:
        return float(np.linalg.eigvalsh(0.5 * (A + A.T))[-1])
    return float(np.max(np.diag(A) + off.sum(axis=1)))


# ------------------------------------------------------------------ figure
def render_figure(A: np.ndarray, h: float, path: Path, dpi: int = 200) -> None:
    norms = [(1, r"$\ell^1$ (diamond)"), (2, r"$\ell^2$ (circle)"),
             (np.inf, r"$\ell^\infty$ (square)")]
    subs = {1: "1", 2: "2", np.inf: r"\infty"}
    E_half, E_full = expm(0.5 * h * A), expm(h * A)

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 9.2), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.865, bottom=0.105,
                        wspace=0.16, hspace=0.16)

    for col, (p, name) in enumerate(norms):
        B = ball_boundary(p)
        nrm = np.linalg.norm(A, ord=p)
        mu = log_norm(A, p)
        rows = [
            # (axes, shown ball(s), dashed bound scale, value label)
            (axes[0, col], [(A @ B, True)], nrm,
             rf"$\|A\|_{{{subs[p]}}}={nrm:.2f}$"),
            (axes[1, col], [(E_half @ B, False), (E_full @ B, True)],
             float(np.exp(mu * h)),
             rf"$\mu_{{{subs[p]}}}(A)={mu:+.2f}$"),
        ]
        for ax, shapes, scale, value in rows:
            ax.set_facecolor(SURFACE)
            ax.set_aspect("equal")
            for side in ax.spines.values():
                side.set_visible(False)
            lim = 2.75 if ax is axes[0, col] else 1.85
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.axhline(0.0, color=MUTED, lw=0.7)
            ax.axvline(0.0, color=MUTED, lw=0.7)
            ticks = [-2, -1, 1, 2] if lim > 2 else [-1, 1]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.tick_params(colors=MUTED, labelsize=8, length=3)

            ax.plot(B[0], B[1], color=INK, lw=1.2)             # unit ball
            for shape, filled in shapes:
                if filled:
                    ax.fill(shape[0], shape[1], color=BLUE, alpha=0.15, lw=0)
                    ax.plot(shape[0], shape[1], color=BLUE, lw=1.8)
                else:                                          # midway snapshot
                    ax.plot(shape[0], shape[1], color=BLUE, lw=1.0, alpha=0.45)
            ax.plot(scale * B[0], scale * B[1], color=ORANGE, lw=1.8,
                    ls=(0, (5, 3)))                            # containing ball
            ax.text(0.03, 0.97, value, transform=ax.transAxes, ha="left",
                    va="top", fontsize=11, color=INK)
        axes[0, col].set_title(name, fontsize=12, color=INK, pad=10)

    fig.text(0.012, 0.66, r"induced norm: $A\,\cdot$ ball", rotation=90,
             ha="left", va="center", fontsize=11, color=MUTED)
    fig.text(0.012, 0.28, rf"log norm: $e^{{hA}}\cdot$ ball ($h={h:g}$)",
             rotation=90, ha="left", va="center", fontsize=11, color=MUTED)

    handles = [
        plt.Line2D([], [], color=INK, lw=1.2, label="unit ball"),
        plt.Line2D([], [], color=BLUE, lw=1.8, label="mapped / flowed ball"),
        plt.Line2D([], [], color=ORANGE, lw=1.8, ls=(0, (5, 3)),
                   label="smallest containing dilation"),
    ]
    fig.legend(handles=handles, loc="upper center", ncols=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.925), labelcolor=INK)
    fig.suptitle(
        r"Induced norms $\|A\|_p$ vs log norms $\mu_p(A)$ — "
        r"$A = [\,-0.2,\ 2\,;\ -2,\ -0.2\,]$,  $\alpha(A)=-0.2$",
        fontsize=12.5, color=INK, y=0.975,
    )
    fig.text(0.5, 0.055,
             r"top: $\|A\|_p$ is the largest stretch of the unit ball under "
             r"$x \mapsto Ax$.   bottom: $e^{hA}B_p \subseteq "
             r"e^{\mu_p(A)h}B_p$ for all $h\geq 0$ — $\mu_p$ is the initial "
             r"growth rate of the flowed ball.",
             ha="center", fontsize=9.5, color=MUTED)
    fig.text(0.5, 0.022,
             r"the rotation shrinks the $\ell^2$ ball ($\mu_2=\alpha=-0.2$) "
             r"but inflates diamond and square ($\mu_1=\mu_\infty=+1.8$) — "
             r"contraction is norm-dependent, stability is not.",
             ha="center", fontsize=9.5, color=MUTED)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Unit-ball visualization of induced matrix norms and "
                    "logarithmic norms for p = 1, 2, inf."
    )
    parser.add_argument("--outdir", type=Path, default=Path("media"))
    parser.add_argument("--flow-time", type=float, default=0.25,
                        help="flow time h for the log-norm row")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)

    A = A_DEFAULT
    h = args.flow_time
    # self-test: ||e^{hA} x||_p <= e^{mu_p h} on sampled unit vectors, and
    # ||A x||_p <= ||A||_p, for every p
    for p in (1, 2, np.inf):
        B = ball_boundary(p)
        mu = log_norm(A, p)
        assert np.all(np.linalg.norm(A @ B, ord=p, axis=0)
                      <= np.linalg.norm(A, ord=p) + 1e-9)
        assert np.all(np.linalg.norm(expm(h * A) @ B, ord=p, axis=0)
                      <= np.exp(mu * h) + 1e-9), f"mu bound violated, p={p}"
        print(f"p={str(p):>3}:  ||A|| = {np.linalg.norm(A, ord=p):.3f}   "
              f"mu(A) = {mu:+.3f}")
    print("self-test: ||Ax|| <= ||A|| and ||e^(hA)x|| <= e^(mu h)  OK")
    print(f"alpha(A) = {np.max(np.linalg.eigvals(A).real):+.3f}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    png = args.outdir / "induced_log_norms.png"
    print(f"Rendering figure -> {png}")
    render_figure(A, h, png, args.dpi)
    print("Done.")


if __name__ == "__main__":
    main()
