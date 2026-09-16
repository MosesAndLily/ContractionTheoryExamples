"""Contraction of the firing-rate model is uniform in the bias.

Same system as ``plot.py``: x' = -x + tanh(Wx + b(t)) with the skew
coupling W = [[0, 0.9], [-0.9, 0]]. The certificate

    osLip(f(., b)) = sup_x mu_inf(-I + [d]W) <= -1 + mu_inf(W) = -0.1

never mentions b: the bias shifts WHERE the Jacobian is evaluated, not
its bound. Consequently the incremental dynamics contract at rate 0.1
for ANY bias, constant or time-varying. Here b(t) switches between four
values every 4 s: every trajectory chases the currently-active
equilibrium x*(b_k) (left), while all pairwise distances decay under
the same envelope e^{-0.1 t}, completely blind to the switches (right)
— entrainment / input-to-state behaviour of contracting systems.

Entry point: ``uv run firing-rate-bias``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from ..style import BLUE, GUIDE, HALO, INK, MUTED, ORANGE, SURFACE
from .plot import RATE, W

BIASES = [np.array(b) for b in
          [(1.0, -1.0), (-1.5, 0.5), (0.5, 1.5), (-0.5, -1.5)]]
T_SWITCH = 4.0
T_END = T_SWITCH * len(BIASES)
N_IC = 6


def equilibrium(b: np.ndarray) -> np.ndarray:
    x = np.zeros(2)
    for _ in range(300):
        x = np.tanh(W @ x + b)
    return x


def simulate(x0: np.ndarray):
    """Integrate through the switching bias, segment by segment."""
    ts, xs = [], []
    x = np.asarray(x0, float)
    for k, b in enumerate(BIASES):
        def f(t, y, b=b):
            return -y + np.tanh(W @ y + b)

        t_eval = np.linspace(k * T_SWITCH, (k + 1) * T_SWITCH, 121)
        sol = solve_ivp(f, (t_eval[0], t_eval[-1]), x, method="DOP853",
                        rtol=1e-12, atol=1e-14, t_eval=t_eval)
        ts.append(sol.t)
        xs.append(sol.y)
        x = sol.y[:, -1]
    return np.concatenate(ts), np.concatenate(xs, axis=1)


# ------------------------------------------------------------------ figure
def render_figure(path: Path, dpi: int = 200) -> None:
    angles = 2.0 * np.pi * np.arange(N_IC) / N_IC
    ics = 2.8 * np.column_stack([np.cos(angles), np.sin(angles)])
    runs = [simulate(x0) for x0 in ics]
    t = runs[0][0]
    ref = runs[0][1]
    envelope = np.exp(-RATE * t)

    # self-test: pairwise contraction blind to the switches
    for _, y in runs[1:]:
        d = np.abs(y - ref).max(axis=0)
        assert np.all(d / d[0] <= envelope * (1.0 + 1e-6)), \
            "distance exceeded the b-independent envelope"

    fig, (ax_s, ax_d) = plt.subplots(
        1, 2, figsize=(12.8, 6.6), dpi=dpi, width_ratios=[1.05, 1.0])
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.83, bottom=0.15,
                        wspace=0.18)

    # ------------------------------ left: state space, moving equilibrium
    ax_s.set_facecolor(SURFACE)
    ax_s.set_aspect("equal")
    for side in ax_s.spines.values():
        side.set_visible(False)
    ax_s.axhline(0.0, color=MUTED, lw=0.8)
    ax_s.axvline(0.0, color=MUTED, lw=0.8)
    ax_s.tick_params(colors=MUTED, labelsize=9)
    ax_s.set_xlim(-3.2, 3.2)
    ax_s.set_ylim(-3.2, 3.2)

    for _, y in runs:
        ax_s.plot(y[0], y[1], color=BLUE, lw=1.0, alpha=0.7, zorder=3)
        ax_s.plot([y[0, 0]], [y[1, 0]], "o", ms=5, mfc="none", mec=BLUE,
                  mew=1.2, zorder=4)
    for k, b in enumerate(BIASES):
        xs = equilibrium(b)
        ax_s.plot([xs[0]], [xs[1]], "o", ms=8, color=ORANGE, zorder=6)
        ax_s.annotate(rf"$x^*(b^{{({k + 1})}})$", xy=xs,
                      xytext=(xs[0] + 0.22, xs[1] + 0.18), fontsize=9.5,
                      color=INK, path_effects=HALO, zorder=6)
    ax_s.set_xlabel(r"$x_1$", fontsize=10, color=INK)
    ax_s.set_ylabel(r"$x_2$", fontsize=10, color=INK)
    ax_s.set_title(
        "the bundle entrains to each bias's equilibrium in turn",
        fontsize=11, color=INK, pad=10)

    # -------------------- right: distances blind to the switching bias
    ax_d.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax_d.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax_d.spines[side].set_color(MUTED)
    ax_d.tick_params(colors=MUTED, labelsize=9)
    ax_d.set_yscale("log")
    ax_d.grid(axis="y", color=GUIDE, lw=0.6, alpha=0.55)
    ax_d.set_axisbelow(True)
    ax_d.set_xlim(0.0, T_END)

    for k in range(len(BIASES)):
        if k % 2 == 1:                       # alternating wash per bias
            ax_d.axvspan(k * T_SWITCH, (k + 1) * T_SWITCH, color=GUIDE,
                         alpha=0.18, lw=0)
        ax_d.text((k + 0.5) * T_SWITCH, 1.45, rf"$b^{{({k + 1})}}$",
                  ha="center", va="bottom", fontsize=9.5, color=MUTED)
    for _, y in runs[1:]:
        d = np.abs(y - ref).max(axis=0)
        ax_d.plot(t, d / d[0], color=BLUE, lw=1.2, alpha=0.75)
    ax_d.plot(t, envelope, color=MUTED, lw=1.6, ls=(0, (5, 3)))
    ax_d.text(13.4, 0.33, r"$e^{-0.1t}$ (certified)", fontsize=9.5,
              color=MUTED, ha="left", va="bottom", path_effects=HALO)
    ax_d.set_ylim(1e-7, 3.2)
    ax_d.set_xlabel("t", fontsize=10, color=INK)
    ax_d.set_ylabel(r"$\|x_i(t)-x_1(t)\|_\infty$ (normalized)",
                    fontsize=10, color=INK)
    ax_d.set_title("pairwise distances never notice the switches",
                   fontsize=11, color=INK, pad=10)

    fig.suptitle(
        r"Contraction is uniform in the bias — $\dot x = -x + "
        rf"\tanh(Wx + b(t))$, $b(t)$ switching every {T_SWITCH:g} s",
        fontsize=12.5, color=INK, y=0.955,
    )
    fig.text(0.5, 0.062,
             r"the certificate $\mathrm{osLip} \leq -1+\mu_\infty(W) = "
             r"-0.1$ never mentions $b$: the bias moves the equilibrium, "
             r"not the contraction rate.",
             ha="center", fontsize=9.5, color=MUTED)
    fig.text(0.5, 0.028,
             r"incremental dynamics $\dot{\delta x} = (\int_0^1 "
             r"Df(y + s\,\delta x,\,b)\,ds)\,\delta x$ — $b$ only "
             r"shifts where $Df$ is evaluated, and the bound is uniform "
             r"in $x$.",
             ha="center", fontsize=9.5, color=MUTED)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Firing-rate model contracting uniformly across a "
                    "switching bias."
    )
    parser.add_argument("--outdir", type=Path, default=Path("media"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)

    for k, b in enumerate(BIASES):
        xs = equilibrium(b)
        print(f"b({k + 1}) = ({b[0]:+.1f}, {b[1]:+.1f})  ->  "
              f"x* = ({xs[0]:+.3f}, {xs[1]:+.3f})")

    args.outdir.mkdir(parents=True, exist_ok=True)
    png = args.outdir / "firing_rate_bias.png"
    print(f"Rendering figure -> {png}")
    render_figure(png, args.dpi)
    print("Done. (envelope self-check ran during rendering)")


if __name__ == "__main__":
    main()
