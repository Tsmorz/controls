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
    ELEVATION = auto()
    AZIMUTH = auto()
    DISTANCE_AZIMUTH_ELEVATION = auto()


class SensorBase(ABC):
    """Abstract base class for sensor measurements."""

    def __init__(
        self,
        state: np.ndarray,
        features: Optional[list[Any]] = None,
        noise: Optional[float] = None,
    ):
        self.pose = state_to_se3(state)

    @abstractmethod
    def as_vector(self) -> np.ndarray:
        """Represent the data as a n-by-1 matrix."""
        pass


class Distance(SensorBase):
    """Construct a distance sensor measurement."""

    def __init__(
        self, state: np.ndarray, features: list[Feature], noise: Optional[float] = None
    ):
        SensorBase.__init__(self, state, features)
        pose = state_to_se3(state)
        dx = np.array([feature.x for feature in features]) - pose.x
        dy = np.array([feature.y for feature in features]) - pose.y
        dz = np.array([feature.z for feature in features]) - pose.z
        distance = np.sqrt(dx**2 + dy**2 + dz**2)

        if noise is not None:
            measurement_noise = np.random.normal(
                loc=0.0,
                scale=noise,
                size=dx.shape,
            )
            distance = distance + measurement_noise

        self.distance: np.ndarray = distance
        self.type = SensorType.DISTANCE

    def as_vector(self) -> np.ndarray:
        """Represent the data as a 1-by-n matrix."""
        return self.distance.reshape((len(self.distance), 1))

    def __str__(self) -> str:
        """Return a string representation of the sensor measurements."""
        distance_str = np.array2string(self.distance, precision=2, floatmode="fixed")
        return f"{self.type.name}(distance:{distance_str})"


class Azimuth(SensorBase):
    """Construct a azimuth sensor measurement."""

    def __init__(
        self, state: np.ndarray, features: list[Feature], noise: Optional[float] = None
    ):
        SensorBase.__init__(self, state, features)
        pose = state_to_se3(state)
        dx = np.array([feature.x for feature in features]) - pose.x
        dy = np.array([feature.y for feature in features]) - pose.y
        dz = np.array([feature.z for feature in features]) - pose.z
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        azimuth = np.arctan2(dy, dx) - pose.yaw

        if noise is not None:
            measurement_noise = np.random.normal(
                loc=0.0,
                scale=noise / (1 + distance),
                size=dx.shape,
            )
            azimuth = azimuth + measurement_noise
        self.azimuth: np.ndarray = azimuth
        self.type = SensorType.AZIMUTH

    def as_vector(self) -> np.ndarray:
        """Represent the data as a 1-by-n matrix."""
        return self.azimuth.reshape((len(self.azimuth), 1))

    def __str__(self) -> str:
        """Return a string representation of the sensor measurements."""
        azimuth_str = np.array2string(self.azimuth, precision=2, floatmode="fixed")
        return f"{self.type.name}(bearing:{azimuth_str})"


class Elevation(SensorBase):
    """Construct a elevation sensor measurement."""

    def __init__(
        self, state: np.ndarray, features: list[Feature], noise: Optional[float] = None
    ):
        SensorBase.__init__(self, state, features)
        pose = state_to_se3(state)
        dx = np.array([feature.x for feature in features]) - pose.x
        dy = np.array([feature.y for feature in features]) - pose.y
        dz = np.array([feature.z for feature in features]) - pose.z
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        elevation = np.arctan2(dz, dx)

        if noise is not None:
            measurement_noise = np.random.normal(
                loc=0.0,
                scale=noise / (1 + distance),
                size=dx.shape,
            )
            elevation = elevation + 0.0 * measurement_noise
        self.elevation: np.ndarray = elevation
        self.type = SensorType.ELEVATION

    def as_vector(self) -> np.ndarray:
        """Represent the data as a 1-by-n matrix."""
        return self.elevation.reshape((len(self.elevation), 1))

    def __str__(self) -> str:
        """Return a string representation of the sensor measurements."""
        elevation_str = np.array2string(self.elevation, precision=2, floatmode="fixed")
        return f"{self.type.name}(elevation:{elevation_str})"


class DistanceAzimuthElevation(Distance, Azimuth, Elevation):
    """Construct a distance and bearing sensor measurement."""

    def __init__(
        self, state: np.ndarray, features: list[Feature], noise: Optional[float] = None
    ):
        Distance.__init__(self, state, features, noise)
        Azimuth.__init__(self, state, features, noise)
        Elevation.__init__(self, state, features, noise)
        self.type = SensorType.DISTANCE_AZIMUTH_ELEVATION

    def as_vector(self) -> np.ndarray:
        """Represent the data as a 1-by-n matrix."""
        merged = np.array((self.distance, self.azimuth)).T.ravel()
        return merged.reshape((len(merged), 1))

    def __str__(self) -> str:
        """Return a string representation of the sensor measurements."""
        distance_str = np.array2string(self.distance, precision=2, floatmode="fixed")
        azimuth_str = np.array2string(self.azimuth, precision=2, floatmode="fixed")
        elevation_str = np.array2string(self.elevation, precision=2, floatmode="fixed")
        return (
            f"{self.type.name}(distance:{distance_str},"
            f" azimuth:{azimuth_str},"
            f" elevation:{elevation_str})"
        )


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

    def as_vector(self) -> np.ndarray:
        """Represent the data as a 1-by-n matrix."""
        return self.state_vec


def get_measurement(
    state: np.ndarray,
    features: list[Any],
    sensor: type[DistanceAzimuthElevation | Distance | Azimuth | Elevation],
) -> DistanceAzimuthElevation | Distance | Azimuth | Elevation:
    """Get a sample measurement for the given sensor.

    :param state: the current state of the system
    :param features: list of features in the map
    :param sensor: the sensor to be used for measurement
    :return: a sensor measurement
    """
    return sensor(state=state, features=features)
