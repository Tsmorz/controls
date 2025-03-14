"""Basic docstring for my module."""

import numpy as np
from jax import numpy as jnp

from config.definitions import (
    DEFAULT_CONTROL,
    LOG_DECIMALS,
)
from src.modules.math_utils import symmetrize_matrix
from src.modules.state_space import StateSpaceNonlinear

jnp.set_printoptions(precision=LOG_DECIMALS)
np.set_printoptions(precision=LOG_DECIMALS)


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
        """Initialize the Kalman Filter.

        :param state_space_nonlinear: nonlinear state space model
        :param process_noise: Process noise covariance
        :param measurement_noise: Measurement noise covariance
        :param initial_covariance: Initial error covariance
        :return: None
        """
        self.ss_nl = state_space_nonlinear
        self.Q: np.ndarray = process_noise
        self.R: np.ndarray = measurement_noise
        self.x: np.ndarray = initial_x
        self.cov: np.ndarray = initial_covariance

    def predict(self, u: np.ndarray = DEFAULT_CONTROL) -> None:
        """Predict the next state and error covariance.

        :param u: Control input
        """
        state_space = self.ss_nl.linearize(self.x, u)

        self.x = state_space.step(x=self.x, u=u)
        self.cov = state_space.A @ self.cov @ state_space.A.T + self.Q
        self.cov = symmetrize_matrix(self.cov)

    def update(self, z: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Update the state estimate with measurement z.

        :param z: Measurement
        :param u: Control input
        :return: Updated state estimate and state covariance
        """
        state_space = self.ss_nl.linearize(self.x, u=u)

        y = z - state_space.C @ self.x  # Measurement residual
        S = state_space.C @ self.cov @ state_space.C.T + self.R  # Innovation covariance
        K = self.cov @ state_space.C.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        cov = (np.eye(self.cov.shape[0]) - K @ state_space.C) @ self.cov
        self.cov = symmetrize_matrix(cov)
        return self.x.copy(), self.cov.copy()
