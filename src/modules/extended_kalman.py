"""Basic docstring for my module."""

from typing import Callable, Optional

import jax
import numpy as np
from loguru import logger

from src.modules.math_utils import symmetrize_matrix
from src.modules.state_space import (
    StateSpace,
)


class ExtendedKalmanFilter:
    """Kalman filter implementation."""

    def __init__(
        self,
        state_space: StateSpace,
        process_noise: np.ndarray,
        measurement_noise: np.ndarray,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
    ) -> None:
        """Initialize the Kalman Filter.

        :param state_space: linear state space model
        :param process_noise: Process noise covariance
        :param measurement_noise: Measurement noise covariance
        :param initial_state: Initial state estimate
        :param initial_covariance: Initial error covariance
        :return: None
        """
        self.state_space = state_space
        self.F: np.ndarray = state_space.A
        self.B: np.ndarray = state_space.B
        self.H: np.ndarray = state_space.C
        self.Q: np.ndarray = process_noise
        self.R: np.ndarray = measurement_noise
        self.cov: np.ndarray = initial_covariance
        self.x: np.ndarray = initial_state

    def predict(self, u: Optional[np.ndarray] = None) -> None:
        """Predict the next state and error covariance.

        :param u: Control input
        """
        if u is None:
            u = np.zeros((self.B.shape[1], 1))
        self.x = self.state_space.step(x=self.x, u=u)

        self.cov = self.F @ self.cov @ self.F.T + self.Q
        self.cov = symmetrize_matrix(self.cov)

    def update(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Update the state estimate with measurement z.

        :param z: Measurement
        :return: Updated state estimate and state covariance
        """
        # print(self.cov_measurement)
        y = z - self.H @ self.x  # Measurement residual
        S = self.H @ self.cov @ self.H.T + self.R  # Innovation covariance
        K = self.cov @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        self.cov = (np.eye(self.cov.shape[0]) - K @ self.H) @ self.cov
        self.cov = symmetrize_matrix(self.cov)

        return self.x.copy(), self.cov.copy()

    def linearize_model(self) -> None:
        """Linearize the state-space model.

        :return: None
        """


def linearize(funcs: list[Callable], x: np.ndarray) -> np.ndarray:
    """Linearize a list of callables.

    :param funcs: List of callables
    :param x: Input points to linearize around
    :return: Jacobian matrix
    """
    jacobian = np.zeros((len(funcs), len(x)))
    for ii, f in enumerate(funcs):
        grad_f = jax.grad(f, argnums=(0, 1))
        jacobian[ii, :] = grad_f(x[0], x[1])
    return jacobian


if __name__ == "__main__":

    def f1(x1, x2) -> float:
        """Create a simple function."""
        value = x1**2 + x1 + 5 + x2
        return value

    def f2(x1, x2) -> float:
        """Create a simple function."""
        value = x2**2 + x2 + 5 + x1
        return value

    jac = linearize([f1, f2], x=np.array([1.0, 1.0]))
    logger.info(jac)
