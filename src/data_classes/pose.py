"""Add a doc string to my files."""

from dataclasses import dataclass


@dataclass
class Pose2D:
    """Represent a two-dimensional pose."""

    x: float
    y: float
    theta: float
