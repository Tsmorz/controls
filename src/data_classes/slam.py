"""Basic docstring for my module."""

from dataclasses import dataclass, field

import numpy as np

from src.data_classes.lie_algebra import SE2
from src.data_classes.map import Feature, Map


@dataclass
class PoseMap:
    """Class representing the pose of the vehicle and the map."""

    pose: SE2 = field(default_factory=SE2)
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
        x, y, theta = ekf_state[0:3, 0]
        self.pose = SE2(x=x, y=y, theta=theta)

        # update the map features
        num_features = len(ekf_state) - 3
        features_x = ekf_state[4::2, 0]
        features_y = ekf_state[5::2, 0]
        features = [
            Feature(x=float(features_x[idx]), y=float(features_y[idx]), id=idx)
            for idx in range(num_features)
        ]
        self.map = Map(features=features)
