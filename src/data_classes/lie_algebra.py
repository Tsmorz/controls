"""Add a doc string to my files."""

import numpy as np
from loguru import logger


class SE2:
    """Represent a two-dimensional pose."""

    def __init__(
        self,
        x: float | np.ndarray = 0.0,
        y: float | np.ndarray = 0.0,
        theta: float | np.ndarray = 0.0,
    ):
        self.x: float | np.ndarray = x
        self.y: float | np.ndarray = y
        self.theta: float | np.ndarray = theta

    def __str__(self):  # pragma: no cover
        """Return a string representation of the pose."""
        msg = f"SE2 Pose=(x:{self.x:.2f}, y:{self.y:.2f}theta:{self.theta:.2f})"
        return msg

    def __matmul__(self, other):
        """Perform a matrix multiplication between two SE2 matrices."""
        if isinstance(other, SE2):
            new_se2 = self.as_matrix() @ other.as_matrix()
            x = new_se2[0, -1]
            y = new_se2[1, -1]
            theta = np.atan2(new_se2[1, 0], new_se2[0, 0])
            return SE2(x=x, y=y, theta=theta)
        else:
            msg = "Matrix multiplication is only supported between SE2 poses."
            logger.error(msg)
            raise ValueError(msg)

    def as_vector(self) -> np.ndarray:
        """Represent the data as a 3-by-1 matrix."""
        return np.array([[self.x], [self.y], [self.theta]])

    def as_matrix(self) -> np.ndarray:
        """Represent the data as a 3-by-3 matrix."""
        matrix = np.array(
            [
                [np.cos(self.theta), -np.sin(self.theta), self.x],
                [np.sin(self.theta), np.cos(self.theta), self.y],
                [0.0, 0.0, 1.0],
            ]
        )
        return matrix
