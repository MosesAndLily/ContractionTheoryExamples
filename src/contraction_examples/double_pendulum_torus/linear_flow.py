"""Linear flow on the configuration torus: the two-bar linkage without
gravity, each joint turning at a constant rate.

With gravity off and joint rates (omega1, omega2), the configuration

    (theta1, theta2) = (theta1_0 + omega1 t, theta2_0 + omega2 t)

is a straight line in the flat torus coordinates — the geodesic flow of
the flat metric on T^2. If omega2/omega1 is irrational the line never
closes and fills the torus densely; a rational ratio p/q closes into a
(q, p) torus curve.

Physics note: for the actual zero-gravity double pendulum the inertial
coupling h = m2 l1 l2 sin(theta1 - theta2) bends the trajectory unless
both joints turn at the same rate (then Delta is constant and h theta_dot^2
terms cancel), so the straight line is exact only along the diagonal. By
default this module drives the linkage kinematically at constant rates
(two decoupled rotors); pass --true-dynamics to integrate the real
zero-gravity double pendulum from the same initial velocities and see the
coupling bend the line.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from ..media_utils import mp4_to_gif, render_video
from ..style import SURFACE
from .animate import OVERSAMPLE, DoublePendulumTorusFigure
from .dynamics import DoublePendulumParams, Trajectory, simulate

GOLDEN_RATIO = (1.0 + np.sqrt(5.0)) / 2.0


def linear_trajectory(
    theta1_0: float, theta2_0: float, omega1: float, omega2: float,
    t_end: float, fps: int,
) -> Trajectory:
    """Constant-rate joint motion: a straight line in (theta1, theta2)."""
    t = np.arange(0.0, t_end, 1.0 / fps)
    return Trajectory(
        t=t,
        theta1=theta1_0 + omega1 * t,
        theta2=theta2_0 + omega2 * t,
        theta1_dot=np.full_like(t, omega1),
        theta2_dot=np.full_like(t, omega2),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render the linear (straight-line) flow of the two-bar "
                    "linkage on its configuration torus."
    )
    parser.add_argument("--outdir", type=Path, default=Path("media"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--theta1-0", type=float, default=0.0)
    parser.add_argument("--theta2-0", type=float, default=0.0)
    parser.add_argument("--omega1", type=float, default=1.0,
                        help="inner joint rate [rad/s]")
    parser.add_argument("--omega2", type=float, default=GOLDEN_RATIO,
                        help="outer joint rate [rad/s]; the default golden "
                             "ratio makes the line wind without closing")
    parser.add_argument("--true-dynamics", action="store_true",
                        help="integrate the real zero-gravity double pendulum "
                             "instead of driving the joints at constant rates")
    parser.add_argument("--t-snap", type=float, default=25.0,
                        help="time of the linkage pose in the summary PNG")
    parser.add_argument("--no-video", action="store_true",
                        help="render only the summary PNG")
    args = parser.parse_args(argv)
    if args.duration * args.fps < 2:
        parser.error("duration must cover at least two frames (duration >= 2/fps)")

    params = DoublePendulumParams(gravity=0.0)
    sample_rate = args.fps * OVERSAMPLE
    if args.true_dynamics:
        traj = simulate(params, args.theta1_0, args.theta2_0,
                        args.omega1, args.omega2, args.duration, sample_rate)
    else:
        traj = linear_trajectory(args.theta1_0, args.theta2_0,
                                 args.omega1, args.omega2,
                                 args.duration, sample_rate)

    def make_figure(dpi: int) -> DoublePendulumTorusFigure:
        return DoublePendulumTorusFigure(
            traj, params, dpi=dpi, show_gravity=False, show_equilibrium=False
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    png = args.outdir / "torus_linear_flow.png"
    print(f"Rendering summary image -> {png}")
    fig_obj = make_figure(dpi=200)
    i_snap = min(int(np.searchsorted(traj.t, args.t_snap)), fig_obj.n - 1)
    fig_obj.draw_summary(i_snap)
    fig_obj.fig.savefig(png, dpi=200, facecolor=SURFACE)
    plt.close(fig_obj.fig)

    if not args.no_video:
        mp4 = args.outdir / "torus_linear_flow.mp4"
        gif = args.outdir / "torus_linear_flow.gif"
        print(f"Rendering video -> {mp4}")
        fig_obj = make_figure(dpi=100)
        render_video(fig_obj.fig, fig_obj.update, fig_obj.n_frames, mp4,
                     fps=args.fps)
        plt.close(fig_obj.fig)
        print(f"Rendering GIF preview -> {gif}")
        mp4_to_gif(mp4, gif)
    print("Done.")


if __name__ == "__main__":
    main()
