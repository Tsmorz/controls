"""Add a doc string to my files."""

from enum import Enum, auto

import numpy as np

from src.data_classes.lie_algebra import SE3
from src.data_classes.map import Feature


class SensorType(Enum):
    """Define the individual sensor types."""

    GPS = auto()
    IMU = auto()
    DISTANCE = auto()
    BEARING = auto()
    DISTANCE_AND_BEARING = auto()


class Distance:
    """Construct a distance sensor measurement."""

    def __init__(self, ground_truth: SE3, features: list[Feature]):
        dx = np.array([feature.x for feature in features]) - ground_truth.x
        dy = np.array([feature.y for feature in features]) - ground_truth.y
        self.distance: np.ndarray = np.sqrt(dx**2 + dy**2)
        self.type = SensorType.DISTANCE

    def as_vector(self) -> np.ndarray:
        """Represent the data as a 1-by-n matrix."""
        return self.distance.reshape((1, len(self.distance)))

    def __str__(self) -> str:
        """Return a string representation of the sensor measurements."""
        distance_str = np.array2string(self.distance, precision=2, floatmode="fixed")
        return f"{self.type.name}(distance:{distance_str})"


class Bearing:
    """Construct a bearing sensor measurement."""

    def __init__(self, ground_truth: SE3, features: list[Feature]):
        dx = np.array([feature.x for feature in features]) - ground_truth.x
        dy = np.array([feature.y for feature in features]) - ground_truth.y
        self.bearing: np.ndarray = np.arctan2(dy, dx) - ground_truth.yaw
        self.type = SensorType.BEARING

    def as_vector(self) -> np.ndarray:
        """Represent the data as a 1-by-n matrix."""
        return self.bearing.reshape((1, len(self.bearing)))

    def __str__(self) -> str:
        """Return a string representation of the sensor measurements."""
        bearing_str = np.array2string(self.bearing, precision=2, floatmode="fixed")
        return f"{self.type.name}(bearing:{bearing_str})"


class DistanceAndBearing(Distance, Bearing):
    """Construct a distance and bearing sensor measurement."""

    def __init__(self, ground_truth: SE3, features: list[Feature]):
        Distance.__init__(self, ground_truth, features)
        Bearing.__init__(self, ground_truth, features)
        self.type = SensorType.DISTANCE_AND_BEARING

    def as_vector(self) -> np.ndarray:
        """Represent the data as a 1-by-n matrix."""
        distances = self.distance.reshape((1, len(self.distance)))
        bearings = self.bearing.reshape((1, len(self.bearing)))
        return np.vstack((distances, bearings))

    def __str__(self) -> str:
        """Return a string representation of the sensor measurements."""
        distance_str = np.array2string(self.distance, precision=2, floatmode="fixed")
        bearing_str = np.array2string(self.bearing, precision=2, floatmode="fixed")
        return f"{self.type.name}(distance:{distance_str}, bearing:{bearing_str})"
