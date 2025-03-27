"""Add a doc string to my files."""

from enum import Enum, auto
from typing import Optional

import numpy as np

from config.definitions import DELTA_T
from src.data_classes.lie_algebra import state_to_se3
from src.data_classes.map import Feature, distance_to_features


def get_azimuth(
    state: np.ndarray,
    features: list[Feature],
    noise: Optional[np.ndarray | float] = None,
) -> np.ndarray:
    """Calculate the azimuth angle from a given pose to a list of features.

    :param state: the current state vector
    :param features: the list of features
    :param noise: optional noise vector for measurement
    :return: the azimuth angles
    """
    pose = state_to_se3(state)
    dx, dy, dz = distance_to_features(pose=pose, features=features)
    distance = np.linalg.norm(np.array([dx, dy, dz]), axis=0)
    azimuth = np.arctan2(dy, dx) - pose.yaw
    if noise is not None:
        measurement_noise = np.random.normal(
            loc=0.0, scale=noise / (1 + distance), size=dx.shape
        )
        azimuth = azimuth + measurement_noise

    return azimuth.reshape((len(azimuth), 1))


def get_elevation(
    state: np.ndarray,
    features: list[Feature],
    noise: Optional[np.ndarray | float] = None,
) -> np.ndarray:
    """Calculate the elevation angle from a given pose to a list of features.

    :param state: the current state vector
    :param features: the list of features
    :param noise: optional noise vector for measurement
    :return: the azimuth angles
    """
    pose = state_to_se3(state)
    dx, dy, dz = distance_to_features(pose=pose, features=features)
    distance = np.linalg.norm(np.array([dx, dy, dz]), axis=0)
    elevation = np.arctan2(dz, dx) - pose.yaw
    if noise is not None:
        measurement_noise = np.random.normal(
            loc=0.0, scale=noise / (1 + distance), size=dx.shape
        )
        elevation = elevation + measurement_noise

    return elevation.reshape((len(elevation), 1))


def get_distance(
    state: np.ndarray,
    features: list[Feature],
    noise: Optional[np.ndarray | float] = None,
) -> np.ndarray:
    """Calculate the distance from a given pose to a list of features.

    :param state: the current state vector
    :param features: the list of features
    :param noise: optional noise vector for measurement
    :return: the distance
    """
    pose = state_to_se3(state)
    dx, dy, dz = distance_to_features(pose=pose, features=features)
    distance = np.linalg.norm(np.array([dx, dy, dz]), axis=0)
    if noise is not None:
        measurement_noise = np.random.normal(loc=0.0, scale=noise, size=dx.shape)
        distance = distance + measurement_noise

    return distance.reshape((len(distance), 1))


def get_distance_azimuth_elevation(
    state: np.ndarray,
    features: list[Feature],
    noise: Optional[np.ndarray | float] = None,
) -> np.ndarray:
    """Calculate the elevation angle from a given pose to a list of features.

    :param state: the current state vector
    :param features: the list of features
    :param noise: optional noise vector for measurement
    :return: the azimuth angles
    """
    distance = get_distance(state=state, features=features, noise=noise)
    azimuth = get_azimuth(state=state, features=features, noise=noise)
    elevation = get_elevation(state=state, features=features, noise=noise)

    merged = np.array((distance, azimuth, elevation)).T.ravel()
    return np.reshape(merged, (len(merged), 1))


def step_dynamics(state_control: np.ndarray, dt: float = DELTA_T) -> np.ndarray:
    """Define the equations of motion.

    :param state_control: the state and control vectors
    :param dt: the time step
    :return: the state vector after applying the motion equations
    """
    vel, omega = state_control[-2:]
    dt_vel, dt_omega = vel[0] * dt, omega[0] * dt
    pose = state_to_se3(state_control[:6, 0])

    state_vec = np.zeros((6, 1))
    state_vec[0, 0] = pose.x + dt_vel * np.cos(pose.yaw) * np.cos(pose.pitch)
    state_vec[1, 0] = pose.y + dt_vel * np.sin(pose.yaw) * np.cos(pose.pitch)
    state_vec[2, 0] = pose.z + dt_vel * np.sin(pose.pitch)
    state_vec[3, 0] = pose.roll
    state_vec[4, 0] = pose.pitch
    state_vec[5, 0] = pose.yaw + dt_omega
    return state_vec


class SensorType(Enum):
    """Define the individual sensor types."""

    GPS = auto()
    IMU = auto()
    DISTANCE = get_distance
    ELEVATION = get_elevation
    AZIMUTH = get_azimuth
    DISTANCE_AZIMUTH_ELEVATION = get_distance_azimuth_elevation
