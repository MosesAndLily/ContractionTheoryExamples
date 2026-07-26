"""Embedding of the configuration torus T^2 = S^1 x S^1 into R^3.

theta1 (inner link angle) runs around the main ring, theta2 (outer link
angle) around the tube. The angular convention matches the pendulum
drawing: theta1 = 0 maps to the front of the ring (facing the camera at
azimuth -90 degrees) and theta2 = 0 to the outer equator, so the hanging
equilibrium (0, 0) sits front and center.
"""

from __future__ import annotations

import numpy as np

RING_RADIUS = 1.0   # R: center of the tube to the torus axis
TUBE_RADIUS = 0.45  # r: tube thickness


def embed(
    theta1: np.ndarray, theta2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map configurations (theta1, theta2) to points on the torus surface."""
    rho = RING_RADIUS + TUBE_RADIUS * np.cos(theta2)
    return rho * np.sin(theta1), -rho * np.cos(theta1), TUBE_RADIUS * np.sin(theta2)


def normal(
    theta1: np.ndarray, theta2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Outward unit normal of the torus surface at (theta1, theta2)."""
    return (
        np.cos(theta2) * np.sin(theta1),
        -np.cos(theta2) * np.cos(theta1),
        np.sin(theta2),
    )


def meridian(theta1: float, n: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tube cross-section circle at a fixed ring angle theta1."""
    theta2 = np.linspace(-np.pi, np.pi, n)
    return embed(np.full_like(theta2, theta1), theta2)


def parallel_circle(theta2: float, n: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Circle around the main ring at a fixed tube angle theta2."""
    theta1 = np.linspace(-np.pi, np.pi, n)
    return embed(theta1, np.full_like(theta1, theta2))
