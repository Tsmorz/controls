"""Basic docstring for my module."""

import argparse

import numpy as np

from config.definitions import MEASUREMENT_NOISE, PROCESS_NOISE
from src.modules.kalman import KalmanFilter, get_measurement
from src.modules.state_space import (
    StateSpace,
    StateSpaceData,
    mass_spring_damper_model,
)


class ExtendedKalmanFilter(KalmanFilter):
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
        super().__init__(
            state_space,
            process_noise,
            measurement_noise,
            initial_state,
            initial_covariance,
        )
        self.state_space = state_space
        self.F: np.ndarray = state_space.A
        self.B: np.ndarray = state_space.B
        self.H: np.ndarray = state_space.C
        self.Q: np.ndarray = process_noise
        self.R: np.ndarray = measurement_noise
        self.cov: np.ndarray = initial_covariance
        self.x: np.ndarray = initial_state


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
    time = np.arange(0, 5, dt).tolist()
    ss = mass_spring_damper_model(discretization_dt=dt)

    kf = ExtendedKalmanFilter(
        state_space=ss,
        process_noise=PROCESS_NOISE * np.eye(2),
        measurement_noise=MEASUREMENT_NOISE * np.eye(2),
        initial_state=np.array([[1.0], [1.0]]),
        initial_covariance=np.eye(2),
    )

    # Generate random control inputs and measurements and update the Kalman filter
    results = StateSpaceData()
    ground_truth_state = kf.x
    for t in time:
        # calculate the feedback control
        gain_matrix = -np.array([[1.0, 1.0]])
        control = gain_matrix @ kf.x

        # Store the updated state for plotting
        results.append_step(t=t, x=kf.x, cov=kf.cov, u=control)

        # step through the filter
        kf.predict(u=control)
        ground_truth_state = ss.step(x=ground_truth_state, u=control)
        measurement = get_measurement(ss.C, state=ground_truth_state, noise=kf.R)
        kf.update(z=measurement)

    ss.plot_history(history=results)
