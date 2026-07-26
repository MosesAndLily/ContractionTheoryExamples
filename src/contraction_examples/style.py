"""Shared plotting style: validated categorical palette (light mode) and
label conventions used by every example figure."""

from __future__ import annotations

from matplotlib import patheffects

BLUE = "#2a78d6"    # trajectories in state/configuration space
ORANGE = "#eb6834"  # the current state (bobs + moving point)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GUIDE = "#c9c8c3"

# white casing so labels stay readable when a trajectory crosses them
HALO = [patheffects.withStroke(linewidth=2.5, foreground=SURFACE)]
