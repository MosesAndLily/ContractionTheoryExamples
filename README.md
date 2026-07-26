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

## Repository layout

```
├── pyproject.toml                     # uv project; console scripts (see [project.scripts])
├── media/                             # rendered outputs, kept in the repo
└── src/contraction_examples/
    └── pendulum_cylinder/
        ├── dynamics.py                # pendulum ODE + integration
        ├── cylinder.py                # embedding of TS¹ ≅ S¹ × ℝ into R³
        ├── animate.py                 # two-panel figure, animation, CLI
        └── lagrangian.py              # Lagrangian colormap on TS¹, CLI
```

Conventions for adding a new example: put the model in its own package with a
`dynamics.py` (simulation only, no plotting), keep geometry/embedding helpers
separate from figure code, expose a console script in `pyproject.toml`, and
render into `media/`.

## Requirements

- [uv](https://docs.astral.sh/uv/) (Python 3.12 is pinned via `.python-version`)
- No system ffmpeg needed — the MP4/GIF are written with the bundled
  `imageio-ffmpeg` binary.
