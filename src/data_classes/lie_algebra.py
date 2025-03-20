"""Add a doc string to my files."""

from typing import Optional

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as Rot

from config.definitions import EULER_ORDER


class SE3:
    """Represent a two-dimensional pose."""

    def __init__(
        self,
        xyz: Optional[np.ndarray] = None,
        roll_pitch_yaw: Optional[np.ndarray] = None,
    ):
        if xyz is None:
            xyz = np.array([[0.0], [0.0], [0.0]])
        if xyz.shape == (3,):
            xyz = np.reshape(xyz, (3, 1))
        if roll_pitch_yaw is None:
            roll_pitch_yaw = np.array([[0.0], [0.0], [0.0]])
        if roll_pitch_yaw.shape == (3,):
            roll_pitch_yaw = np.reshape(roll_pitch_yaw, (3, 1))
        self.x: float | np.ndarray = xyz[0, 0]
        self.y: float | np.ndarray = xyz[1, 0]
        self.z: float | np.ndarray = xyz[2, 0]
        self.roll: float | np.ndarray = roll_pitch_yaw[0, 0]
        self.pitch: float | np.ndarray = roll_pitch_yaw[1, 0]
        self.yaw: float | np.ndarray = roll_pitch_yaw[2, 0]

    def __str__(self):  # pragma: no cover
        """Return a string representation of the pose."""
        msg = (
            f"SE3 Pose=(x:{self.x:.2f}, y:{self.y:.2f}, z:{self.z:.2f}, "
            f"roll:{self.roll:.2f}, pitch:{self.pitch:.2f}, yaw:{self.yaw:.2f})"
        )
        return msg

    def __matmul__(self, other):
        """Perform a matrix multiplication between two SE2 matrices."""
        if isinstance(other, SE3):
            dim = (3, 1)
            new_se3 = self.as_matrix() @ other.as_matrix()
            xyz = new_se3[:3, -1]
            rpy = Rot.from_matrix(matrix=new_se3[:3, :3]).as_euler(
                EULER_ORDER, degrees=False
            )
            return SE3(xyz=np.reshape(xyz, dim), roll_pitch_yaw=np.reshape(rpy, dim))
        else:
            msg = "Matrix multiplication is only supported between SE2 poses."
            logger.error(msg)
            raise ValueError(msg)

    def as_vector(self) -> np.ndarray:
        """Represent the data as a 3-by-1 matrix."""
        return np.array(
            [[self.x], [self.y], [self.z], [self.roll], [self.pitch], [self.yaw]]
        )

    def as_matrix(self) -> np.ndarray:
        """Represent the data as a 3-by-3 matrix."""
        rpy = np.array([self.roll, self.pitch, self.yaw])
        rot = Rot.from_euler(angles=rpy, seq="XYZ", degrees=False).as_matrix()

        trans = np.array([[self.x], [self.y], [self.z]])

        matrix = np.hstack((rot, trans))
        matrix = np.vstack((matrix, np.array([[0.0, 0.0, 0.0, 1.0]])))
        return matrix
