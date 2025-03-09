"""Basic docstring for my module."""

import argparse
from typing import Optional

import numpy as np
from definitions import NUM_STEPS


class KalmanFilter:
    """Kalman filter implementation."""

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
    ) -> None:
        """Initialize the Kalman Filter.

        :param A: State transition matrix
        :param B: Control input matrix
        :param C: Observation matrix
        :param Q: Process noise covariance
        :param R: Measurement noise covariance
        :param initial_state: Initial state estimate
        :param initial_covariance: Initial error covariance
        :return: None
        """
        self.A: np.ndarray = A
        self.B: np.ndarray = B
        self.H: np.ndarray = C
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
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        y = z - self.H @ self.x  # Measurement residual
        self.x = self.x + K @ y
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P

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

    A = np.array(
        [
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )
    B = np.array(
        [
            [0.5],
            [1.0],
        ]
    )
    C = np.array(
        [
            [1, 0],
            [0, 1],
        ]
    )  # Observation matrix
    Q = np.array(
        [
            [1e-4, 0],
            [0, 1e-4],
        ]
    )  # Process noise covariance
    R = np.array([[1e-2]])  # Measurement noise covariance
    P = np.eye(2)  # Initial error covariance
    x = np.array(
        [
            [0],
            [0],
        ]
    )  # Initial state estimate

    kf = KalmanFilter(A, B, C, Q, R, initial_state=x, initial_covariance=P)
    for _ in range(NUM_STEPS):
        m = np.random.rand(2, 1)
        kf.predict(u=np.array([[1]]))
        kf.update(z=m)
    print(kf.history)


if __name__ == "__main__":  # pragma: no cover
    main()
