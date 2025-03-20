"""Add a doc string to my files."""

from typing import Optional

import numpy as np
import numpy.typing as npt

from config.definitions import DEFAULT_UNITS, DEFAULT_VARIANCE


class SE2:
    """Represent a two-dimensional pose."""

    def __init__(
        self,
        x: float | np.ndarray = 0.0,
        y: float | np.ndarray = 0.0,
        theta: float | np.ndarray = 0.0,
        covariance: Optional[np.ndarray] = None,
        units: str = DEFAULT_UNITS,
    ):
        self.x: float | np.ndarray = x
        self.y: float | np.ndarray = y
        self.theta: float | np.ndarray = theta
        if covariance is None:
            covariance = DEFAULT_VARIANCE * np.eye(2)
        self.covariance: np.ndarray = covariance
        self.units: str = units

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
