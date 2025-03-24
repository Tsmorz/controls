"""Add a doc string to my files."""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Optional

import numpy as np

from src.data_classes.lie_algebra import state_to_se3
from src.data_classes.map import Feature


class SensorType(Enum):
    """Define the individual sensor types."""

    GPS = auto()
    IMU = auto()
    DISTANCE = auto()
    BEARING = auto()
    DISTANCE_AND_BEARING = auto()


class SensorBase(ABC):
    """Abstract base class for sensor measurements."""

    def __init__(self, state: np.ndarray, features: Optional[list[Any]] = None):
        self.pose = state_to_se3(state)

    @abstractmethod
    def as_vector(self) -> np.ndarray:
        """Represent the data as a n-by-1 matrix."""
        pass


class Distance(SensorBase):
    """Construct a distance sensor measurement."""

    def __init__(self, state: np.ndarray, features: list[Feature]):
        SensorBase.__init__(self, state, features)
        pose = state_to_se3(state)
        dx = np.array([feature.x for feature in features]) - pose.x
        dy = np.array([feature.y for feature in features]) - pose.y
        self.distance: np.ndarray = np.sqrt(dx**2 + dy**2)
        self.type = SensorType.DISTANCE
        self.dim = 1

    def as_vector(self) -> np.ndarray:
        """Represent the data as a 1-by-n matrix."""
        return self.distance.reshape((len(self.distance), 1))

    def __str__(self) -> str:
        """Return a string representation of the sensor measurements."""
        distance_str = np.array2string(self.distance, precision=2, floatmode="fixed")
        return f"{self.type.name}(distance:{distance_str})"


class Bearing(SensorBase):
    """Construct a bearing sensor measurement."""

    def __init__(self, state: np.ndarray, features: list[Feature]):
        SensorBase.__init__(self, state, features)
        pose = state_to_se3(state)
        dx = np.array([feature.x for feature in features]) - pose.x
        dy = np.array([feature.y for feature in features]) - pose.y
        self.bearing: np.ndarray = np.arctan2(dy, dx) - pose.yaw
        self.type = SensorType.BEARING
        self.dim = 1

    def as_vector(self) -> np.ndarray:
        """Represent the data as a 1-by-n matrix."""
        return self.bearing.reshape((len(self.bearing), 1))

    def __str__(self) -> str:
        """Return a string representation of the sensor measurements."""
        bearing_str = np.array2string(self.bearing, precision=2, floatmode="fixed")
        return f"{self.type.name}(bearing:{bearing_str})"


class DistanceAndBearing(Distance, Bearing):
    """Construct a distance and bearing sensor measurement."""

    def __init__(self, state: np.ndarray, features: list[Feature]):
        Distance.__init__(self, state, features)
        Bearing.__init__(self, state, features)
        self.type = SensorType.DISTANCE_AND_BEARING
        self.dim = 2

    def as_vector(self) -> np.ndarray:
        """Represent the data as a 1-by-n matrix."""
        merged = np.array((self.distance, self.bearing)).T.ravel()
        return merged.reshape((len(merged), 1))

    def __str__(self) -> str:
        """Return a string representation of the sensor measurements."""
        distance_str = np.array2string(self.distance, precision=2, floatmode="fixed")
        bearing_str = np.array2string(self.bearing, precision=2, floatmode="fixed")
        return f"{self.type.name}(distance:{distance_str}, bearing:{bearing_str})"


class Dynamics:
    """Create an object for the system dynamics."""

    def __init__(self, state_control: np.ndarray):
        """Define the equations of motion.

        :param state_control: the state and control vectors
        :return: the state vector after applying the motion equations
        """
        vel, omega = state_control[-2:]
        pose = state_to_se3(state_control[:6, 0])

        state_vec = np.zeros((6, 1))
        state_vec[0, 0] = vel[0] * np.cos(pose.yaw) * np.cos(pose.pitch) + pose.x
        state_vec[1, 0] = vel[0] * np.sin(pose.yaw) * np.cos(pose.pitch) + pose.y
        state_vec[2, 0] = vel[0] * np.sin(pose.pitch) + pose.z
        state_vec[3, 0] = pose.roll
        state_vec[4, 0] = pose.pitch
        state_vec[5, 0] = pose.yaw + omega[0]
        self.state_vec = state_vec
        self.dim = 6

    def as_vector(self) -> np.ndarray:
        """Represent the data as a 1-by-n matrix."""
        return self.state_vec


def get_measurement(
    state: np.ndarray,
    features: list[Any],
    sensor: type[DistanceAndBearing | Distance | Bearing],
) -> DistanceAndBearing | Distance | Bearing:
    """Get a sample measurement for the given sensor.

    :param state: the current state of the system
    :param features: list of features in the map
    :param sensor: the sensor to be used for measurement
    :return: a sensor measurement
    """
    return sensor(state=state, features=features)
