"""Add a doc string to my files."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

from config.definitions import DEFAULT_UNITS


@dataclass
class SE2:
    """Represent a two-dimensional pose."""

    x: float | np.ndarray
    y: float | np.ndarray
    theta: float | np.ndarray
    covariance: Optional[np.ndarray] = None
    units: Optional[str] = DEFAULT_UNITS

    def __str__(self):  # pragma: no cover
        """Return a string representation of the pose."""
        msg = (
            f"SE2 Pose=(x:{self.x:.2f} {self.units}, "
            f"y:{self.y:.2f} {self.units}, "
            f"theta:{self.theta:.2f} deg)"
        )
        return msg

    def as_vector(self) -> npt.NDArray[np.float64]:
        """Represent the data as a 3-by-1 matrix."""
        return np.asarray(
            np.array([[self.x], [self.y], [self.theta]]), dtype=np.float64
        )

    def as_matrix(self) -> np.ndarray:
        """Represent the data as a 3-by-3 matrix."""
        matrix = np.ones(3)
        matrix[:2, :2] = np.array(
            [
                [np.cos(self.theta), -np.sin(self.theta), self.x],
                [np.sin(self.theta), np.cos(self.theta), self.y],
                [0.0, 0.0, 1.0],
            ]
        )
        return matrix


@dataclass
class Velocity:
    """Represent the velocity as a dataclass."""

    x: float | np.ndarray
    y: float | np.ndarray
