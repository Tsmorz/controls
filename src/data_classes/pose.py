"""Add a doc string to my files."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Pose2D:
    """Represent a two-dimensional pose."""

    x: float | np.ndarray
    y: float | np.ndarray
    theta: float | np.ndarray
    covariance: Optional[np.ndarray] = None

    def __str__(self):  # pragma: no cover
        """Return a string representation of the pose."""
        return f"Pose2D(x:{self.x:.2f}, y:{self.y:.2f}, theta:{self.theta:.2f})"

    def as_vector(self) -> np.ndarray:
        return np.array([[self.x], [self.y], [self.theta]])


@dataclass
class Pose3D:
    """Represent a three-dimensional pose."""

    xyz: np.ndarray
    rot_matrix: np.ndarray
