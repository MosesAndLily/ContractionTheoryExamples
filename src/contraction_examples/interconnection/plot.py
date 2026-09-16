"""Feedback interconnection of two contracting subsystems — the
"better combination": no common metric needed, each block keeps its own
norm, and the whole network is certified by a 2x2 Metzler eigenvalue.

    x1' = A1 x1 + B12 x2        mu(A1) <= -lam1,  ||B12|| <= gam12
    x2' = B21 x1 + A2 x2        mu(A2) <= -lam2,  ||B21|| <= gam21

Block norms obey the comparison inequalities

    D+||dx1|| <= -lam1 ||dx1|| + gam12 ||dx2||
    D+||dx2|| <=  gam21 ||dx1|| - lam2 ||dx2||

so the VECTOR of block distances is dominated by the Metzler gain matrix

    Gamma = [[-lam1, gam12], [gam21, -lam2]].

If Gamma is Hurwitz (<=> lam1 lam2 > gam12 gam21, the small-gain
condition), its Perron vector eta > 0 turns the composite weighted norm

    V(dx) = max( ||dx1||/eta1, ||dx2||/eta2 )

into a contraction certificate: D+ V <= alpha(Gamma) V < 0. A 2x2
eigenvalue test certifies the full network, at rate |alpha(Gamma)|.

Instance: A1 = -I + 2J, A2 = -0.5 I + J (J = rotation), B12 = 1.2k I,
B21 = 0.6k I. Certificate threshold k* = sqrt(0.5/0.72) = 0.833; the
true network destabilizes at k ~ 1.0 (mild conservatism, from ignoring
coupling phase). Entry point: ``uv run feedback-interconnection``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from ..style import BLUE, GUIDE, HALO, INK, MUTED, ORANGE, SURFACE

J = np.array([[0.0, 1.0], [-1.0, 0.0]])
A1 = -1.0 * np.eye(2) + 2.0 * J        # mu_2(A1) = -1.0
A2 = -0.5 * np.eye(2) + 1.0 * J        # mu_2(A2) = -0.5
LAM = (1.0, 0.5)
GAM = (1.2, 0.6)                       # coupling gains per unit k
K_STAR = float(np.sqrt(LAM[0] * LAM[1] / (GAM[0] * GAM[1])))
K_DEMO = 0.6


def full(k: float) -> np.ndarray:
    return np.block([[A1, GAM[0] * k * np.eye(2)],
                     [GAM[1] * k * np.eye(2), A2]])


def gain(k: float) -> np.ndarray:
    return np.array([[-LAM[0], GAM[0] * k], [GAM[1] * k, -LAM[1]]])


def alpha(M: np.ndarray) -> float:
    return float(np.max(np.linalg.eigvals(M).real))


def perron(G: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eig(G)
    eta = np.abs(v[:, np.argmax(w.real)])
    return eta / eta[1]


# ------------------------------------------------------------------ figure
def render_figure(path: Path, dpi: int = 200) -> None:
    ks = np.linspace(0.0, 1.6, 401)
    a_gain = np.array([alpha(gain(k)) for k in ks])
    a_true = np.array([alpha(full(k)) for k in ks])

    G = gain(K_DEMO)
    rate = -alpha(G)                       # 0.183 at k = 0.6
    eta = perron(G)
    t = np.linspace(0.0, 20.0, 401)
    rng = np.random.default_rng(3)
    d0s = rng.normal(size=(6, 4))
    d0s /= np.linalg.norm(d0s, axis=1, keepdims=True)
    Phi = [expm(full(K_DEMO) * ti) for ti in t]

    def vnorm(d: np.ndarray) -> float:
        return max(np.linalg.norm(d[:2]) / eta[0],
                   np.linalg.norm(d[2:]) / eta[1])

    fig, (ax_k, ax_d) = plt.subplots(1, 2, figsize=(12.6, 6.2), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.815, bottom=0.16,
                        wspace=0.22)
    for ax in (ax_k, ax_d):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)

    # ------------- left: 2x2 Metzler certificate vs true 4x4 abscissa
    ax_k.axhline(0.0, color=INK, lw=1.0)
    ax_k.axvspan(0.0, K_STAR, color=GUIDE, alpha=0.22, lw=0)
    ax_k.plot(ks, a_gain, color=ORANGE, lw=2.0)
    ax_k.plot(ks, a_true, color=BLUE, lw=2.0)
    ax_k.axvline(K_STAR, color=ORANGE, lw=1.2, ls=(0, (4, 3)))
    ax_k.text(K_STAR - 0.03, -0.62, r"$k^*=0.833$" "\ncertificate limit",
              ha="right", fontsize=9, color=ORANGE, path_effects=HALO)
    ax_k.text(0.35, -0.10, "certified\ncontracting", ha="center",
              fontsize=9, color=MUTED, path_effects=HALO)
    ax_k.text(1.24, -0.28, r"true $\alpha(A(k))$", fontsize=9.5,
              color=BLUE, path_effects=HALO)
    ax_k.text(1.24, 0.45, r"$\alpha(\Gamma(k))$ (2$\times$2 test)",
              fontsize=9.5, color=ORANGE, path_effects=HALO)
    ax_k.axvline(K_DEMO, color=MUTED, lw=1.0, ls=(0, (2, 2)))
    ax_k.text(K_DEMO + 0.02, -0.68, r"$k=0.6$ (right panel)", fontsize=8.5,
              color=MUTED, path_effects=HALO)
    ax_k.set_xlim(0.0, 1.6)
    ax_k.set_ylim(-0.75, 0.72)
    ax_k.set_xlabel("coupling strength  $k$", fontsize=10, color=INK)
    ax_k.set_ylabel("growth rate", fontsize=10, color=INK)
    ax_k.set_title(
        r"a $2\times 2$ Metzler eigenvalue certifies the $4\times 4$ "
        "network\n"
        r"$\Gamma(k) = [\,-\lambda_1,\ \gamma_{12}k\,;\ \gamma_{21}k,"
        r"\ -\lambda_2\,]$",
        fontsize=10.5, color=INK, pad=8)

    # ------- right: composite Perron-weighted norm decays at rate 0.183
    ax_d.set_yscale("log")
    ax_d.grid(axis="y", color=GUIDE, lw=0.6, alpha=0.55)
    ax_d.set_axisbelow(True)
    for d0 in d0s:
        v = np.array([vnorm(Ph @ d0) for Ph in Phi])
        ax_d.plot(t, v / v[0], color=BLUE, lw=1.2, alpha=0.75)
        assert np.all(v / v[0] <= np.exp(-rate * t) * (1 + 1e-9)), \
            "composite-norm envelope violated"
    ax_d.plot(t, np.exp(-rate * t), color=MUTED, lw=1.6, ls=(0, (5, 3)))
    ax_d.text(15.1, np.exp(-rate * 15.1) * 1.35,
              rf"$e^{{\alpha(\Gamma)t}} = e^{{-{rate:.3f}t}}$",
              fontsize=9.5, color=MUTED, ha="left", path_effects=HALO)
    ax_d.set_xlim(0.0, 20.0)
    ax_d.set_ylim(2e-3, 3.0)
    ax_d.set_xlabel("t", fontsize=10, color=INK)
    ax_d.set_ylabel(r"$V(\delta x(t))\,/\,V(\delta x(0))$", fontsize=10,
                    color=INK)
    ax_d.set_title(
        rf"$k={K_DEMO}$: composite norm $V=\max(\|\delta x_1\|/"
        rf"{eta[0]:.2f},\ \|\delta x_2\|)$" "\n"
        r"decays under the certified envelope ($\eta$ = Perron vector of "
        r"$\Gamma$)",
        fontsize=10.5, color=INK, pad=8)

    fig.suptitle(
        r"Feedback interconnection of contracting subsystems — each block "
        r"keeps its own metric; only the gains couple",
        fontsize=12.5, color=INK, y=0.96,
    )
    fig.text(0.5, 0.065,
             r"$D^+\|\delta x_i\| \leq -\lambda_i\|\delta x_i\| + "
             r"\gamma_{ij}\|\delta x_j\|$: block distances are dominated "
             r"by the Metzler matrix $\Gamma$ — Hurwitz iff "
             r"$\lambda_1\lambda_2 > \gamma_{12}\gamma_{21}$ (small gain).",
             ha="center", fontsize=9.5, color=MUTED)
    fig.text(0.5, 0.028,
             r"$\lambda_1=1$, $\lambda_2=0.5$, $\gamma_{12}=1.2k$, "
             r"$\gamma_{21}=0.6k$; certificate ignores coupling phase, so "
             r"the true network survives to $k\approx 1.0 > k^*$.",
             ha="center", fontsize=9.5, color=MUTED)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Network contraction of a two-block feedback "
                    "interconnection via the Metzler gain-matrix test."
    )
    parser.add_argument("--outdir", type=Path, default=Path("media"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)

    # self-tests: block rates, the threshold algebra, and conservatism
    assert abs(alpha(A1) + 1.0) < 1e-12 and abs(alpha(A2) + 0.5) < 1e-12
    assert abs(alpha(gain(K_STAR))) < 1e-12, "threshold algebra broken"
    assert alpha(full(K_STAR)) < 0.0, "true network should outlive k*"
    G = gain(K_DEMO)
    eta = perron(G)
    print(f"k* = {K_STAR:.4f};  at k = {K_DEMO}: alpha(Gamma) = "
          f"{alpha(G):+.4f}, Perron eta = ({eta[0]:.3f}, {eta[1]:.3f})")
    print(f"true alpha(A) at k*: {alpha(full(K_STAR)):+.3f}  "
          f"(certificate is conservative, as expected)")

    args.outdir.mkdir(parents=True, exist_ok=True)
    png = args.outdir / "feedback_interconnection.png"
    print(f"Rendering figure -> {png}")
    render_figure(png, args.dpi)
    print("Done. (envelope self-check ran during rendering)")


if __name__ == "__main__":
    main()
