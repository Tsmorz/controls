"""Basic docstring for my module."""

import argparse
from typing import Optional

import numpy as np
from loguru import logger
from state_space import StateSpace

from config.definitions import DEFAULT_NUM_STEPS, DEFAULT_VARIANCE


class KalmanFilter:
    """Kalman filter implementation."""

    def __init__(
        self,
        state_space: StateSpace,
        Q: np.ndarray,
        R: np.ndarray,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
    ) -> None:
        """Initialize the Kalman Filter.

        :param state_space: linear state space model
        :param Q: Process noise covariance
        :param R: Measurement noise covariance
        :param initial_state: Initial state estimate
        :param initial_covariance: Initial error covariance
        :return: None
        """
        self.state_space = state_space
        self.A: np.ndarray = state_space.A
        self.B: np.ndarray = state_space.B
        self.C: np.ndarray = state_space.C
        self.cov_process: np.ndarray = Q
        self.cov_meas: np.ndarray = R
        self.cov: np.ndarray = initial_covariance
        self.x: np.ndarray = initial_state
        self.history: list[tuple[np.ndarray, np.ndarray]] = [(self.x, self.cov)]

    def predict(self, u: Optional[np.ndarray] = None) -> None:
        """Predict the next state and error covariance.

        :param u: Control input
        """
        if u is None:
            u = np.zeros((self.B.shape[1], 1))

        self.x = self.state_space.step(self.x, u)
        self.cov = self.A @ self.cov @ self.A.T + self.cov_process

    def update(self, z: np.ndarray) -> None:
        """Update the state estimate with measurement z.

        :param z: Measurement
        """
        S = self.C @ self.cov @ self.C.T + self.cov_meas  # Innovation covariance
        K = self.cov @ self.C.T @ np.linalg.inv(S)  # Kalman gain
        y = z - self.C @ self.x  # Measurement residual
        self.x = self.x + K @ y
        self.cov = (np.eye(self.cov.shape[0]) - K @ self.C) @ self.cov

        # Store the updated state for plotting
        self.history.append((self.x.copy(), self.cov.copy()))


if __name__ == "__main__":  # pragma: no cover
    """Run the main program with this function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dir",
        action="store",
        default=None,
        help="Directory to process.",
    )

    ss = StateSpace(
        A=np.array([[1.0, 1.0], [0.0, 1.0]]),
        B=np.array([[0.5], [1.0]]),
        C=np.array([[1, 0], [0, 1]]),
    )

    x0 = np.array([[0], [0]])
    kf = KalmanFilter(
        state_space=ss,
        Q=DEFAULT_VARIANCE * np.eye(2),
        R=DEFAULT_VARIANCE * np.eye(1),
        initial_state=x0,
        initial_covariance=np.eye(2),
    )

    # Generate random control inputs and measurements and update the Kalman filter
    for _ in range(DEFAULT_NUM_STEPS):
        m = np.random.rand(2, 1)
        kf.predict(u=np.array([[1]]))
        kf.update(z=m)
    logger.info(kf.history)
