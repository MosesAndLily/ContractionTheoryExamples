"""Parallel combination of LTI systems under a COMMON contraction metric.

The slide's idea, made precise: if two systems contract in the SAME
metric P,

    A_1' P + P A_1 <= -2 lam_1 P,     A_2' P + P A_2 <= -2 lam_2 P,

then multiplying by c_1, c_2 >= 0 and adding gives, for the parallel
combination A = c_1 A_1 + c_2 A_2,

    A' P + P A <= -2 (c_1 lam_1 + c_2 lam_2) P,

i.e. mu_P(c_1 A_1 + c_2 A_2) <= c_1 mu_P(A_1) + c_2 mu_P(A_2) — the
log norm is sublinear, so contraction in a FIXED metric is closed under
conic combination, with the rates combining linearly.

The common metric is essential, and the figure shows both sides:

  left  — C_1 = [[-1,10],[0,-1]] and C_2 = C_1' are each Hurwitz
          (alpha = -1) yet share no metric: the mixture
          theta C_1 + (1-theta) C_2 is UNSTABLE for almost every theta
          (alpha = -1 + 10 sqrt(theta(1-theta)), peak +4).
  right — A_1 = [[-0.8,1],[-4,-0.2]], A_2 = [[-0.5,-0.6],[0.8,-1]]
          share P = diag(4, 1): mu_P(A_1) = -0.20, mu_P(A_2) = -0.28,
          and every mixture sits on or below the chord (sublinearity),
          strictly negative. (In the identity metric mu_2(A_1) = +1.03
          — the SHARED metric, not stability alone, is what combines.)

Entry point: ``uv run parallel-combination``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..style import BLUE, GUIDE, HALO, INK, MUTED, ORANGE, SURFACE

T = np.diag([2.0, 1.0])                 # P^{1/2}; the shared metric P = T@T
P = T @ T
A1 = np.linalg.inv(T) @ np.array([[-0.8, 2.0], [-2.0, -0.2]]) @ T
A2 = np.linalg.inv(T) @ np.array([[-0.5, -1.2], [0.4, -1.0]]) @ T
C1 = np.array([[-1.0, 10.0], [0.0, -1.0]])  # Hurwitz pair with NO shared P
C2 = C1.T


def mu_P(A: np.ndarray) -> float:
    """Log norm in the P-metric: mu_2(T A T^{-1})."""
    M = T @ A @ np.linalg.inv(T)
    return float(np.linalg.eigvalsh(0.5 * (M + M.T))[-1])


def alpha(A: np.ndarray) -> float:
    return float(np.max(np.linalg.eigvals(A).real))


# ------------------------------------------------------------------ figure
def render_figure(path: Path, dpi: int = 200) -> None:
    th = np.linspace(0.0, 1.0, 401)
    mix = lambda M1, M2: [t * M1 + (1.0 - t) * M2 for t in th]
    a_bad = np.array([alpha(M) for M in mix(C1, C2)])
    m_good = np.array([mu_P(M) for M in mix(A1, A2)])
    mu1, mu2_ = mu_P(A1), mu_P(A2)
    chord = th * mu1 + (1.0 - th) * mu2_

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.6, 6.2), dpi=dpi,
                                     sharex=True)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.815, bottom=0.16,
                        wspace=0.20)

    for ax in (ax_l, ax_r):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.axhline(0.0, color=INK, lw=1.0)
        ax.set_xlabel(r"$\theta$   ($c_1=\theta$, $c_2=1-\theta$)",
                      fontsize=10, color=INK)

    # ------------- left: Hurwitz alone does not survive the combination
    ax_l.fill_between(th, 0.0, np.maximum(a_bad, 0.0), color=ORANGE,
                      alpha=0.12, lw=0)
    ax_l.plot(th, a_bad, color=ORANGE, lw=2.0)
    for t0, y0 in [(0.0, -1.0), (1.0, -1.0)]:
        ax_l.plot([t0], [y0], "o", ms=7, mfc="none", mec=ORANGE, mew=1.8)
    ax_l.text(0.5, 4.15, "unstable mixtures", ha="center", va="bottom",
              fontsize=9.5, color=ORANGE, path_effects=HALO)
    ax_l.text(0.03, -1.35, r"$\alpha(C_1)=\alpha(C_2)=-1$ (each Hurwitz)",
              ha="left", fontsize=9, color=INK, path_effects=HALO)
    ax_l.set_ylim(-2.1, 5.2)
    ax_l.set_ylabel(r"$\alpha(\theta C_1 + (1-\theta)C_2)$", fontsize=10,
                    color=INK)
    ax_l.set_title(
        "no shared metric: stability does not combine\n"
        r"$C_1=[\,-1,\ 10\,;\ 0,\ -1\,]$,  $C_2=C_1^{\top}$",
        fontsize=10.5, color=INK, pad=8)

    # ------------- right: a shared metric P makes combination convex
    ax_r.plot(th, chord, color=MUTED, lw=1.6, ls=(0, (5, 3)))
    ax_r.plot(th, m_good, color=BLUE, lw=2.0)
    for t0, y0 in [(0.0, mu2_), (1.0, mu1)]:
        ax_r.plot([t0], [y0], "o", ms=7, mfc="none", mec=BLUE, mew=1.8)
    ax_r.text(0.50, chord[200] + 0.022,
              r"chord $\theta\mu_P(A_1)+(1-\theta)\mu_P(A_2)$",
              ha="center", va="bottom", fontsize=9, color=MUTED,
              path_effects=HALO)
    ax_r.text(0.50, m_good[200] - 0.022, r"$\mu_P(\theta A_1+(1-\theta)A_2)$",
              ha="center", va="top", fontsize=9.5, color=BLUE,
              path_effects=HALO)
    ax_r.text(0.03, 0.028, "unstable side", ha="left", va="bottom",
              fontsize=8.5, color=MUTED)
    ax_r.set_ylim(-0.55, 0.12)
    ax_r.set_ylabel(r"$\mu_P(\theta A_1 + (1-\theta)A_2)$", fontsize=10,
                    color=INK)
    ax_r.set_title(
        "shared metric $P=\\mathrm{diag}(4,1)$: every mixture contracts\n"
        r"$A_1=[\,-0.8,\ 1\,;\ -4,\ -0.2\,]$,  "
        r"$A_2=[\,-0.5,\ -0.6\,;\ 0.8,\ -1\,]$",
        fontsize=10.5, color=INK, pad=8)

    fig.suptitle(
        r"Parallel combination $\dot x = (c_1 A_1 + c_2 A_2)\,x$ — "
        r"contraction in a common metric is closed under mixing",
        fontsize=12.5, color=INK, y=0.96,
    )
    fig.text(0.5, 0.065,
             r"$A_i^{\top}P + PA_i \preceq -2\lambda_i P$ scaled by "
             r"$c_i \geq 0$ and summed: $A^{\top}P + PA \preceq "
             r"-2(c_1\lambda_1 + c_2\lambda_2)P$ — rates mix linearly, "
             r"in the SAME $P$.",
             ha="center", fontsize=9.5, color=MUTED)
    fig.text(0.5, 0.028,
             r"equivalently $\mu_P(c_1A_1+c_2A_2) \leq c_1\mu_P(A_1) + "
             r"c_2\mu_P(A_2)$ (sublinearity); note $\mu_2(A_1)=+1.03$: "
             r"the identity metric certifies neither endpoint.",
             ha="center", fontsize=9.5, color=MUTED)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Parallel combination of LTI systems: contraction in a "
                    "common metric is convex; bare stability is not."
    )
    parser.add_argument("--outdir", type=Path, default=Path("media"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)

    mu1, mu2_ = mu_P(A1), mu_P(A2)
    # self-test 1: the Lyapunov LMIs hold in the shared metric
    for A, lam in [(A1, -mu1), (A2, -mu2_)]:
        S = A.T @ P + P @ A + 2.0 * lam * P
        assert np.max(np.linalg.eigvalsh(0.5 * (S + S.T))) <= 1e-9, \
            "shared-metric LMI violated"
    # self-test 2: sublinearity along the whole segment
    th = np.linspace(0.0, 1.0, 401)
    for t in th:
        assert mu_P(t * A1 + (1 - t) * A2) <= t * mu1 + (1 - t) * mu2_ + 1e-9
    # self-test 3: the counterexample behaves as advertised
    assert alpha(C1) < 0 and alpha(C2) < 0
    assert alpha(0.5 * C1 + 0.5 * C2) > 0

    print(f"mu_P(A1) = {mu1:+.3f}, mu_P(A2) = {mu2_:+.3f}  (P = diag(4,1))")
    print(f"worst mixture: mu_P <= {max(mu_P(t*A1+(1-t)*A2) for t in th):+.3f}")
    print(f"counterexample: alpha(C1) = alpha(C2) = -1, "
          f"alpha(midpoint) = {alpha(0.5*C1+0.5*C2):+.1f}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    png = args.outdir / "parallel_combination.png"
    print(f"Rendering figure -> {png}")
    render_figure(png, args.dpi)
    print("Done. (LMI, sublinearity, and counterexample self-tests passed)")


if __name__ == "__main__":
    main()
