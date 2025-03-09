"""Basic docstring for my module."""

import argparse
from typing import Optional

import numpy as np
from definitions import NUM_STEPS
from state_space import StateSpace


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
        self.A: np.ndarray = state_space.A
        self.B: np.ndarray = state_space.B
        self.C: np.ndarray = state_space.C
        self.Q: np.ndarray = Q
        self.R: np.ndarray = R
        self.P: np.ndarray = initial_covariance
        self.x: np.ndarray = initial_state
        self.history: list[np.ndarray] = [self.x]

    def predict(self, u: Optional[np.ndarray] = None) -> None:
        """Predict the next state and error covariance.

        :param u: Control input
        """
        if u is None:
            u = np.zeros((self.B.shape[1], 1))

        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z: np.ndarray) -> None:
        """Update the state estimate with measurement z.

        :param z: Measurement
        """
        S = self.C @ self.P @ self.C.T + self.R  # Innovation covariance
        K = self.P @ self.C.T @ np.linalg.inv(S)  # Kalman gain
        y = z - self.C @ self.x  # Measurement residual
        self.x = self.x + K @ y
        self.P = (np.eye(self.P.shape[0]) - K @ self.C) @ self.P

        self.history.append(self.x.copy())  # Store the updated state for plotting

    def get_state(self) -> np.ndarray:
        """Get the current state estimate.

        :return: Current state estimate
        """
        return self.x


def main():
    """Run the main program with this function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dir",
        action="store",
        default=None,
        help="Directory to process.",
    )

    state_space = StateSpace(
        A=np.array([[1.0, 1.0], [0.0, 1.0]]),
        B=np.array([[0.5], [1.0]]),
        C=np.array([[1, 0], [0, 1]]),
    )
    # Process noise covariance
    Q = np.array(
        [
            [1e-4, 0],
            [0, 1e-4],
        ]
    )
    # Measurement noise covariance
    R = np.array([[1e-2]])

    kf = KalmanFilter(
        state_space,
        Q,
        R,
        initial_state=np.array([[0], [0]]),
        initial_covariance=np.eye(2),
    )

    # Generate random control inputs and measurements and update the Kalman filter
    for _ in range(NUM_STEPS):
        m = np.random.rand(2, 1)
        kf.predict(u=np.array([[1]]))
        kf.update(z=m)
    print(kf.history)


if __name__ == "__main__":  # pragma: no cover
    main()
