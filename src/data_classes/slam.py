"""Basic docstring for my module."""

from dataclasses import dataclass, field

import numpy as np

from src.data_classes.lie_algebra import SE3
from src.data_classes.map import Feature, Map


@dataclass
class PoseMap:
    """Class representing the pose of the vehicle and the map."""

    pose: SE3 = field(default_factory=SE3)
    map: Map = field(default_factory=Map)

    def as_vector(self) -> np.ndarray:
        """Return the state data as a vector."""
        vector = np.zeros((3 + 2 * len(self.map.features), 1))
        return vector

    def from_vector(self, ekf_state: np.ndarray) -> None:
        """Convert the state vector to the corresponding attributes.

        :param ekf_state: The state vector to convert
        :return: None
        """
        # update the pose
        xyz = ekf_state[0:3]
        rpy = ekf_state[3:6]
        self.pose = SE3(xyz=xyz, roll_pitch_yaw=rpy)

        # update the map features
        num_features = len(ekf_state) - 6
        features_x = ekf_state[6::2, 0]
        features_y = ekf_state[7::2, 0]
        features = [
            Feature(x=float(features_x[idx]), y=float(features_y[idx]), id=idx)
            for idx in range(num_features)
        ]
        self.map = Map(features=features)
