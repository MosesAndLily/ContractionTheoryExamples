"""Embedding of the tangent bundle TS^1 = S^1 x R as a cylinder in R^3.

The base coordinate theta wraps around the unit circle and the fiber
coordinate theta_dot runs along the cylinder axis. The angular convention
matches the pendulum drawing: theta = 0 (hanging down) maps to (0, -1, 0),
so the stable equilibrium faces the camera at azimuth -90 degrees.
"""

from __future__ import annotations

import numpy as np

RADIUS = 1.0


def embed(
    theta: np.ndarray, theta_dot: np.ndarray, velocity_scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map states (theta, theta_dot) to points on the cylinder x^2 + y^2 = R^2."""
    x = RADIUS * np.sin(theta)
    y = -RADIUS * np.cos(theta)
    z = theta_dot / velocity_scale
    return x, y, z


def surface_mesh(
    z_min: float, z_max: float, n_theta: int = 100, n_z: int = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mesh of the cylinder surface for a translucent backdrop."""
    theta = np.linspace(-np.pi, np.pi, n_theta)
    z = np.linspace(z_min, z_max, n_z)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x = RADIUS * np.sin(theta_grid)
    y = -RADIUS * np.cos(theta_grid)
    return x, y, z_grid


def rim_circle(z: float, n: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Horizontal circle on the cylinder at a fixed fiber height z."""
    theta = np.linspace(-np.pi, np.pi, n)
    return RADIUS * np.sin(theta), -RADIUS * np.cos(theta), np.full_like(theta, z)
