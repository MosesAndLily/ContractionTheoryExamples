# Contraction Theory Examples

Animated examples for contraction theory and geometric mechanics. Each example
lives in its own package under `src/contraction_examples/` and renders its
media (MP4 / GIF / PNG) into `media/` via a small CLI.

## Example 1 — a pendulum and its tangent-bundle cylinder

A damped simple pendulum (left) next to the geometric picture of its dynamics
(right): the configuration is an angle θ on the circle S¹, so the full state
(θ, θ̇) lives on the tangent bundle

$$TS^1 \cong S^1 \times \mathbb{R},$$

which is a cylinder — θ wraps around it, θ̇ runs along its axis. The pendulum
starts with enough velocity to go over the top a few times (the trajectory
*wraps* around the cylinder — no ±π jump, which is exactly why the cylinder and
not the plane is the right state space), then damping captures it and the
trajectory spirals into the stable equilibrium (θ, θ̇) = (0, 0).

![Pendulum and its state trajectory on the cylinder](media/pendulum_cylinder.gif)

Summary figure (full trajectory, pendulum shown mid-swing):
[`media/pendulum_cylinder.png`](media/pendulum_cylinder.png) · video:
[`media/pendulum_cylinder.mp4`](media/pendulum_cylinder.mp4)

### Model

$$\ddot{\theta} = -\frac{g}{L}\sin\theta - c\,\dot{\theta},
\qquad g = 9.81\ \mathrm{m/s^2},\ L = 1\ \mathrm{m},\ c = 0.55\ \mathrm{s^{-1}},$$

integrated with `scipy.integrate.solve_ivp` (DOP853) from
(θ₀, θ̇₀) = (−π/2, 11 rad/s).

### The Lagrangian on the tangent bundle

The pendulum's Lagrangian (per unit mass and rod length)

$$\mathcal{L}(\theta, \dot{\theta})/ml^2 = \tfrac{1}{2}\dot{\theta}^2 + \frac{g}{l}\cos\theta$$

as a colormap over the state space — flat (θ, θ̇) chart on the left, painted
onto the cylinder on the right — with the same damped trajectory drawn on top
of it. On the flat chart the trajectory jumps at the ±π seam; on the cylinder
it just wraps.

![Lagrangian colormap on the cylinder with the trajectory](media/lagrangian_cylinder.png)

### Run it

```sh
uv sync
uv run pendulum-cylinder            # renders PNG + MP4 + GIF into media/
uv run pendulum-cylinder --no-video # summary PNG only (fast)
uv run pendulum-lagrangian          # Lagrangian colormap + trajectory PNG
```

Useful flags for `pendulum-cylinder`: `--duration`, `--fps`, `--damping`,
`--theta0`, `--theta-dot0`, `--t-snap` (pendulum pose in the summary PNG),
`--outdir`. For `pendulum-lagrangian`: the same simulation flags plus
`--no-trajectory` for the bare colormap. See `--help` on either command.

## Example 2 — a double pendulum and its configuration torus

A planar double pendulum in absolute angles (θ₁, θ₂), each measured from the
downward vertical. The configuration space is the torus

$$T^2 \cong S^1 \times S^1,$$

drawn as a wireframe: θ₁ runs around the main ring, θ₂ around the tube. The
undamped chaotic motion (energy is conserved to ~1e-9 by the DOP853
integrator) makes the configuration point wind around both directions of the
torus, leaving a persistent trace with the recent past drawn brighter.

![Double pendulum and its configuration trace on the torus](media/double_pendulum_torus.gif)

Summary figure: [`media/double_pendulum_torus.png`](media/double_pendulum_torus.png)
· video: [`media/double_pendulum_torus.mp4`](media/double_pendulum_torus.mp4)

```sh
uv run double-pendulum-torus            # renders PNG + MP4 + GIF into media/
uv run double-pendulum-torus --no-video # summary PNG only (fast)
```

Flags: `--duration`, `--fps`, `--theta1-0`, `--theta2-0`, `--theta1-dot0`,
`--theta2-dot0`, `--damping1`, `--damping2`, `--t-snap`, `--outdir`.

### Linear flow — a straight line on the torus

The same two-bar linkage with gravity off and each joint turning at a
constant rate ω₁, ω₂: the configuration traces a *straight line* in the flat
torus coordinates, (θ₁, θ₂) = (θ₁₀ + ω₁t, θ₂₀ + ω₂t) — the geodesic flow of
the flat metric on T². With ω₂/ω₁ = φ (the golden ratio, default) the line
never closes and fills the torus densely; a rational ratio closes into a
torus knot/link curve.

![Linear flow of the linkage on the torus](media/torus_linear_flow.gif)

Summary figure: [`media/torus_linear_flow.png`](media/torus_linear_flow.png)
· video: [`media/torus_linear_flow.mp4`](media/torus_linear_flow.mp4)

```sh
uv run torus-linear-flow                      # golden-ratio winding
uv run torus-linear-flow --omega2 1.5         # closes into a (2,3) curve
uv run torus-linear-flow --true-dynamics      # real zero-g double pendulum
```

Physics note: for the *actual* zero-gravity double pendulum, constant joint
rates solve the equations of motion exactly only when ω₁ = ω₂ (the inertial
coupling term m₂l₁l₂ sin(θ₁−θ₂)·θ̇² cancels); otherwise the coupling bends
the line. The default drives the joints kinematically (two decoupled
rotors); `--true-dynamics` integrates the real thing from the same initial
velocities so you can see the difference.

## Repository layout

```
├── pyproject.toml                     # uv project; console scripts (see [project.scripts])
├── media/                             # rendered outputs, kept in the repo
└── src/contraction_examples/
    ├── style.py                       # shared palette + label conventions
    ├── media_utils.py                 # shared MP4/GIF writers (bundled ffmpeg)
    ├── pendulum_cylinder/
    │   ├── dynamics.py                # pendulum ODE + integration
    │   ├── cylinder.py                # embedding of TS¹ ≅ S¹ × ℝ into R³
    │   ├── animate.py                 # two-panel figure, animation, CLI
    │   └── lagrangian.py              # Lagrangian colormap on TS¹, CLI
    └── double_pendulum_torus/
        ├── dynamics.py                # double pendulum ODE + energy + integration
        ├── torus.py                   # embedding of T² ≅ S¹ × S¹ into R³
        ├── animate.py                 # two-panel figure, animation, CLI
        └── linear_flow.py             # straight-line (geodesic) flow on T², CLI
```

Conventions for adding a new example: put the model in its own package with a
`dynamics.py` (simulation only, no plotting), keep geometry/embedding helpers
separate from figure code, reuse `style.py` (palette) and `media_utils.py`
(MP4/GIF writers), expose a console script in `pyproject.toml`, and render
into `media/`.

## Requirements

- [uv](https://docs.astral.sh/uv/) (Python 3.12 is pinned via `.python-version`)
- No system ffmpeg needed — the MP4/GIF are written with the bundled
  `imageio-ffmpeg` binary.
