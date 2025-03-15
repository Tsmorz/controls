"""Basic docstring for my module."""

from typing import Any, Optional

import numpy as np

from config.definitions import (
    DEFAULT_CONTROL,
)
from src.data_classes.map import Feature
from src.data_classes.pose import Pose2D
from src.modules.math_utils import symmetrize_matrix
from src.modules.state_space import StateSpaceLinear, StateSpaceNonlinear


class ExtendedKalmanFilter:
    """Kalman filter implementation."""

    def __init__(
        self,
        state_space_nonlinear: StateSpaceNonlinear,
        process_noise: np.ndarray,
        measurement_noise: np.ndarray,
        initial_x: np.ndarray,
        initial_covariance: np.ndarray,
    ) -> None:
        """Initialize the Extended Kalman Filter.

        :param state_space_nonlinear: nonlinear state space model
        :param process_noise: Process noise covariance
        :param measurement_noise: Measurement noise covariance
        :param initial_x: Initial state estimate
        :param initial_covariance: Initial error covariance
        :return: None
        """
        self.state_space_nl = state_space_nonlinear
        self.Q: np.ndarray = process_noise
        self.R: np.ndarray = measurement_noise
        self.x: np.ndarray = initial_x
        self.cov: np.ndarray = initial_covariance

    def predict(self, u: np.ndarray = DEFAULT_CONTROL) -> None:
        """Predict the next state and error covariance.

        :param u: Control input
        """
        A, B = self.state_space_nl.linearize(
            model=self.state_space_nl.motion_model, x=self.x, u=u
        )
        state_space = StateSpaceLinear(A, B)

        self.x = self.state_space_nl.step(x=self.x, u=u)
        self.cov = state_space.A @ self.cov @ state_space.A.T + self.Q
        self.cov = symmetrize_matrix(self.cov)

    def update(
        self, z: np.ndarray, u: np.ndarray, other_args: Optional[Any] = None
    ) -> None:
        """Update the state estimate with measurement z.

        :param z: Measurement
        :param u: Control input
        :param other_args: Additional arguments (e.g., map of features)
        :return: Updated state estimate and state covariance
        """
        A, B = self.state_space_nl.linearize(
            model=self.state_space_nl.motion_model, x=self.x, u=u
        )
        C, D = self.state_space_nl.linearize(
            model=self.state_space_nl.measurement_model,
            x=self.x,
            u=u,
            other_args=other_args,
        )
        state_space = StateSpaceLinear(A, B, C, D)

        y = z - self.state_space_nl.predict_z(
            self.x, u, other_args
        )  # Measurement residual
        S = state_space.C @ self.cov @ state_space.C.T + self.R  # Innovation covariance
        K = self.cov @ state_space.C.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        cov = (np.eye(self.cov.shape[0]) - K @ state_space.C) @ self.cov
        self.cov = symmetrize_matrix(cov)


def get_range_distance_measurement(pose: Pose2D, feature: Feature) -> np.ndarray:
    """Calculate the range distance measurement between a pose and a feature.

    :param pose: Pose of the robot
    :param feature: Feature in the map
    :return: Range distance measurement
    """
    delta_x = feature.x - pose.x
    delta_y = feature.y - pose.y
    distance = np.sqrt(delta_x**2 + delta_y**2)

    angle = np.arctan(delta_y / delta_x) - pose.theta

    return np.array([[distance], [angle]])
