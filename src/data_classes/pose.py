"""Add a doc string to my files."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Pose2D:
    """Represent a two-dimensional pose."""

    x: float
    y: float
    theta: float


@dataclass
class Pose3D:
    """Represent a three-dimensional pose."""

    xyz: np.ndarray
    rot_matrix: np.ndarray
