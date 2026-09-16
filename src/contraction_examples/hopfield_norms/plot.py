"""Contraction of an excitatory Hopfield network in its natural norm.

Dynamics x' = -x + W tanh(x) with W = [[0, 1.6], [0.4, 0]] >= 0. Every
unweighted norm fails the certificate ||W||_p < 1 (all equal 1.60), but
W >= 0 makes the Jacobian J = -I + W diag(tanh') Metzler, and for
Metzler Jacobians the RIGHT norm is dictated by the structure itself:
weight the ell-inf norm by the Perron vector v of W (here W v = 0.8 v
with v = (2, 1)):

    ||dx||       = max( |dx_1|/2, |dx_2| )
    mu(J(x))    <= -1 + ||diag(1/v) W diag(v)||_inf = -1 + rho(W) = -0.2

so the network contracts at certified rate 1 - rho(W) = 0.2.

Left panel: a fan of trajectories launched on the boundary of the
weighted ball of radius 0.9 around a reference trajectory; the ball
(a 2:1 box, the shape of v) shrinks as 0.9 e^{-0.2 t} and provably
traps the whole fan forever. Right panel: every pairwise distance in
the weighted norm decays monotonically under the certified envelope,
while the same distance in the unweighted ell-inf norm overshoots —
the wrong norm cannot certify what the right norm makes obvious.

Entry point: ``uv run hopfield-norms``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.integrate import solve_ivp

from ..style import BLUE, GUIDE, HALO, INK, MUTED, ORANGE, SURFACE

AQUA = "#1baf7a"  # categorical slot 3: the (unsuitable) unweighted ell-inf

W = np.array([[0.0, 1.6], [0.4, 0.0]])
V = np.array([2.0, 1.0])       # Perron vector: W V = 0.8 V
RATE = 0.2                     # certified rate 1 - rho(W)
R0 = 0.9                       # initial weighted-ball radius
REF0 = np.array([0.4, 0.2])
OFFSETS = [(1, 1), (1, -1), (-1, 1), (-1, -1),
           (1, 0), (-1, 0), (0, 1), (0, -1)]   # boundary of the ball
BOX_TIMES = [0.0, 1.5, 3.0, 5.0, 8.0]
T_END = 10.0


def wnorm(d: np.ndarray) -> np.ndarray:
    """Perron-weighted ell-inf norm, columnwise."""
    return np.abs(d / V[:, None]).max(axis=0)


def simulate(x0: np.ndarray):
    def f(t, x):
        return -x + W @ np.tanh(x)

    t_eval = np.linspace(0.0, T_END, 501)
    sol = solve_ivp(f, (0.0, T_END), x0, method="DOP853",
                    rtol=1e-9, atol=1e-12, t_eval=t_eval)
    return t_eval, sol.y


# ------------------------------------------------------------------ figure
def render_figure(path: Path, dpi: int = 200) -> None:
    t, ref = simulate(REF0)
    fan = [simulate(REF0 + R0 * np.array(s) * V)[1] for s in OFFSETS]
    envelope = R0 * np.exp(-RATE * t)
    # self-test: the certified shrinking ball traps every fan member
    for y in fan:
        assert np.all(wnorm(y - ref) <= envelope * (1.0 + 1e-6)), \
            "certified weighted ball violated"

    fig, (ax_s, ax_d) = plt.subplots(
        1, 2, figsize=(12.8, 6.6), dpi=dpi, width_ratios=[1.15, 1.0])
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.825, bottom=0.155,
                        wspace=0.18)

    # -------------------- left: state space, the shrinking weighted ball
    ax_s.set_facecolor(SURFACE)
    ax_s.set_aspect("equal")
    for side in ax_s.spines.values():
        side.set_visible(False)
    ax_s.axhline(0.0, color=MUTED, lw=0.8)
    ax_s.axvline(0.0, color=MUTED, lw=0.8)
    ax_s.tick_params(colors=MUTED, labelsize=9)

    for y in fan:
        ax_s.plot(y[0], y[1], color=BLUE, lw=1.0, alpha=0.75, zorder=3)
        ax_s.plot([y[0, 0]], [y[1, 0]], "o", ms=4.5, mfc="none", mec=BLUE,
                  mew=1.1, zorder=3)
    ax_s.plot(ref[0], ref[1], color=INK, lw=2.0, zorder=4)
    ax_s.plot([0.0], [0.0], "o", ms=7, color=INK, zorder=6)
    ax_s.text(0.07, -0.13, "equilibrium", fontsize=8.5, color=INK,
              ha="left", va="top", path_effects=HALO)

    for tb in BOX_TIMES:
        i = int(np.argmin(np.abs(t - tb)))
        half = V * R0 * np.exp(-RATE * tb)
        c = ref[:, i]
        ax_s.add_patch(Rectangle(c - half, 2 * half[0], 2 * half[1],
                                 fc="none", ec=ORANGE, lw=1.5,
                                 ls=(0, (5, 3)), zorder=5))
        ax_s.text(c[0] - half[0] + 0.06, c[1] + half[1] - 0.05,
                  rf"$t={tb:g}$", fontsize=8.5, color=ORANGE, ha="left",
                  va="top", path_effects=HALO, zorder=6)
    ax_s.set_xlabel(r"$x_1$", fontsize=10, color=INK)
    ax_s.set_ylabel(r"$x_2$", fontsize=10, color=INK)
    ax_s.set_title(
        r"the weighted ball $\|x-x_{\mathrm{ref}}\|_{\infty,v}\leq"
        rf" {R0:g}\,e^{{-{RATE:g}t}}$ traps the fan",
        fontsize=11, color=INK, pad=10)

    # ---------------- right: certified monotone decay vs the wrong norm
    ax_d.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax_d.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax_d.spines[side].set_color(MUTED)
    ax_d.tick_params(colors=MUTED, labelsize=9)
    ax_d.set_yscale("log")
    ax_d.grid(axis="y", color=GUIDE, lw=0.6, alpha=0.55)
    ax_d.set_axisbelow(True)

    for y in fan:
        ax_d.plot(t, wnorm(y - ref) / R0, color=BLUE, lw=1.1, alpha=0.65)
    ax_d.plot(t, envelope / R0, color=MUTED, lw=1.6, ls=(0, (5, 3)))
    d_bad = fan[OFFSETS.index((0, 1))] - ref     # worst case for ell-inf
    bad = np.abs(d_bad).max(axis=0)
    ax_d.plot(t, bad / bad[0], color=AQUA, lw=2.0)

    ax_d.text(8.35, 0.255, r"$e^{-0.2t}$", fontsize=9.5, color=MUTED,
              ha="left", va="bottom", path_effects=HALO)
    handles = [
        plt.Line2D([], [], color=BLUE, lw=1.4,
                   label=r"$\|\delta x\|_{\infty,v}$ (8 pairs) — monotone"),
        plt.Line2D([], [], color=AQUA, lw=2.0,
                   label=r"unweighted $\ell^\infty$ — overshoots, "
                         r"no certificate"),
        plt.Line2D([], [], color=MUTED, lw=1.6, ls=(0, (5, 3)),
                   label=r"certified envelope $e^{-0.2t}$"),
    ]
    ax_d.legend(handles=handles, loc="lower left", frameon=False,
                fontsize=8.5, labelcolor=INK)
    ax_d.set_xlabel("t", fontsize=10, color=INK)
    ax_d.set_ylabel(r"$\|\delta x(t)\|\,/\,\|\delta x(0)\|$", fontsize=10,
                    color=INK)
    ax_d.set_title("certified decay in the Perron-weighted norm",
                   fontsize=11, color=INK, pad=10)

    fig.suptitle(
        r"Excitatory Hopfield network $\dot x = -x + W\tanh(x)$, "
        r"$W=[\,0,\ 1.6\,;\ 0.4,\ 0\,]\geq 0$ — contraction in the norm "
        r"the structure dictates",
        fontsize=12.5, color=INK, y=0.955,
    )
    fig.text(0.5, 0.075,
             r"$W\geq 0$ makes $J=-I+W\,\mathrm{diag}(\tanh')$ Metzler, so "
             r"the Perron vector $v=(2,1)$ ($Wv=0.8v$) defines "
             r"$\|\delta x\|_{\infty,v}=\max(|\delta x_1|/2,\,|\delta x_2|)$:",
             ha="center", fontsize=9.5, color=MUTED)
    fig.text(0.5, 0.038,
             r"$\mu_{\infty,v}(J(x)) \leq -1+"
             r"\|\mathrm{diag}(1/v)\,W\,\mathrm{diag}(v)\|_\infty"
             r" = -1+\rho(W) = -0.2$ — certified rate $0.2$, although "
             r"$\|W\|_1=\|W\|_2=\|W\|_\infty=1.6$ all fail.",
             ha="center", fontsize=9.5, color=MUTED)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Contraction of an excitatory Hopfield network in the "
                    "Perron-weighted ell-inf norm."
    )
    parser.add_argument("--outdir", type=Path, default=Path("media"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)

    rho = float(np.max(np.abs(np.linalg.eigvals(W))))
    Tw = np.diag(1.0 / V)
    wgt = float(np.linalg.norm(Tw @ W @ np.linalg.inv(Tw), np.inf))
    print(f"||W||_1 = {np.linalg.norm(W, 1):.2f}, "
          f"||W||_2 = {np.linalg.norm(W, 2):.2f}, "
          f"||W||_inf = {np.linalg.norm(W, np.inf):.2f}  (all >= 1: fail)")
    print(f"rho(W) = {rho:.2f},  Perron v = {V},  "
          f"||diag(1/v) W diag(v)||_inf = {wgt:.2f} < 1  "
          f"-> certified rate {1.0 - wgt:.2f}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    png = args.outdir / "hopfield_norms.png"
    print(f"Rendering figure -> {png}")
    render_figure(png, args.dpi)
    print("Done. (ball-containment self-check ran during rendering)")


if __name__ == "__main__":
    main()
