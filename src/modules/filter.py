"""Basic docstring for my module."""

import argparse
from typing import Optional

import numpy as np
from state_space import (
    StateSpace,
    StateSpaceData,
    mass_spring_damper_model,
)

from config.definitions import DEFAULT_VARIANCE


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
        self.cov_measurement: np.ndarray = R
        self.cov: np.ndarray = initial_covariance
        self.x: np.ndarray = initial_state

    def predict(self, u: Optional[np.ndarray] = None) -> None:
        """Predict the next state and error covariance.

        :param u: Control input
        """
        self.x = self.state_space.step(x=self.x, u=u)
        self.cov = self.A @ self.cov @ self.A.T + self.cov_process

    def update(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Update the state estimate with measurement z.

        :param z: Measurement
        :return: Updated state estimate and state covariance
        """
        S = self.C @ self.cov @ self.C.T + self.cov_measurement  # Innovation covariance
        K = self.cov @ self.C.T @ np.linalg.inv(S)  # Kalman gain
        y = z - self.C @ self.x  # Measurement residual
        self.x = self.x + K @ y
        self.cov = (np.eye(self.cov.shape[0]) - K @ self.C) @ self.cov
        return self.x.copy(), self.cov.copy()


def get_measurement(state: np.ndarray, noise: float = DEFAULT_VARIANCE) -> np.ndarray:
    """Get a measurement of the state.

    :param state: True state
    :param noise: Variance of the noise
    :return: Measurement of the state
    """
    state += np.random.normal(loc=0, scale=noise, size=state.shape)
    return state


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

    dt = 0.05
    ss = mass_spring_damper_model(discretization_dt=dt)

    kf = KalmanFilter(
        state_space=ss,
        Q=DEFAULT_VARIANCE * np.eye(2),
        R=DEFAULT_VARIANCE * np.eye(1),
        initial_state=np.array([[1.0], [1.0]]),
        initial_covariance=np.eye(2),
    )

    # Generate random control inputs and measurements and update the Kalman filter
    results = StateSpaceData()
    ground_truth_state = kf.x
    for t in np.arange(0, 10, dt):
        # calculate the feedback control
        gain_matrix = -np.array([[1.0, 1.0]])
        control = gain_matrix @ kf.x

        # Store the updated state for plotting
        results.append_step(t=t, x=kf.x, cov=kf.cov, u=control)

        # keep track of the ground truth
        ground_truth_state = ss.step(x=ground_truth_state, u=control)

        # run the filter
        kf.predict(u=control)
        measurement = get_measurement(state=ground_truth_state)
        kf.update(z=measurement)

    ss.plot_history(history=results)
