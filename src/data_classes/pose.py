"""Add a doc string to my files."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Pose2D:
    """Represent a two-dimensional pose."""

    x: float | np.ndarray
    y: float | np.ndarray
    theta: float | np.ndarray

    def as_vector(self) -> np.ndarray:
        return np.array([[self.x], [self.y], [self.theta]])


@dataclass
class Pose3D:
    """Represent a three-dimensional pose."""

    xyz: np.ndarray
    rot_matrix: np.ndarray
