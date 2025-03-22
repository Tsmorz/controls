"""Basic docstring for my module."""

from typing import Any, Optional

import numpy as np

from config.definitions import DEFAULT_CONTROL, MEASUREMENT_NOISE, PROCESS_NOISE
from src.modules.math_utils import symmetrize_matrix
from src.modules.state_space import StateSpaceLinear, StateSpaceNonlinear


class ExtendedKalmanFilter:
    """Extended Kalman filter implementation."""

    def __init__(
        self,
        state_space_nonlinear: StateSpaceNonlinear,
        initial_x: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise: Optional[np.ndarray] = None,
        measurement_noise: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize the Extended Kalman Filter.

        :param state_space_nonlinear: nonlinear state space model
        :param initial_x: Initial state estimate
        :param initial_covariance: Initial error covariance
        :param process_noise: Process noise covariance
        :param measurement_noise: Measurement noise covariance
        :return: None
        """
        self.state_space_nl = state_space_nonlinear

        if process_noise is None:
            process_noise = PROCESS_NOISE * np.eye(
                len(state_space_nonlinear.motion_model)
            )
        self.Q: np.ndarray = process_noise

        if measurement_noise is None:
            measurement_noise = MEASUREMENT_NOISE * np.eye(
                len(state_space_nonlinear.measurement_model)
            )
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
        self, z: np.ndarray, u: np.ndarray, measurement_args: Optional[Any] = None
    ) -> None:
        """Update the state estimate with measurement z.

        :param z: Measurement
        :param u: Control input
        :param measurement_args: Additional arguments (e.g., map of features)
        :return: Updated state estimate and state covariance
        """
        A, B = self.state_space_nl.linearize(
            model=self.state_space_nl.motion_model, x=self.x, u=u
        )
        C, D = self.state_space_nl.linearize(
            model=self.state_space_nl.measurement_model,
            x=self.x,
            u=u,
            other_args=measurement_args,
        )
        state_space = StateSpaceLinear(A, B, C, D)

        y = z - self.state_space_nl.predict_z(self.x, u, measurement_args)
        S = state_space.C @ self.cov @ state_space.C.T + self.R
        K = self.cov @ state_space.C.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        cov = (np.eye(self.cov.shape[0]) - K @ state_space.C) @ self.cov
        self.cov = symmetrize_matrix(cov)
