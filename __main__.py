"""Basic docstring for my module."""

import argparse

import numpy as np
from loguru import logger

from config.definitions import MEASUREMENT_NOISE, PROCESS_NOISE
from src.modules.controller import full_state_feedback, get_control_input
from src.modules.filter import KalmanFilter, get_measurement
from src.modules.state_space import StateSpaceData, mass_spring_damper_model


def main(dir_path: str) -> None:
    """Pipeline to run the repo code."""
    logger.info(f"Path to directory: {dir_path}")
    dt = 0.05
    time = np.arange(0, 10, dt).tolist()
    ss = mass_spring_damper_model(discretization_dt=dt)

    # find the desired control gains
    desired_eigenvalues = np.array([0.89 + 0.29j, 0.89 - 0.29j])
    gain_matrix = full_state_feedback(ss, desired_eigenvalues)
    desired_state = np.array([[0], [0]])

    # initialize the kalman filter
    kf = KalmanFilter(
        state_space=ss,
        process_noise=PROCESS_NOISE * np.eye(2),
        measurement_noise=MEASUREMENT_NOISE * np.eye(2),
        initial_state=np.array([[5.0], [5.0]]),
        initial_covariance=5 * np.eye(2),
    )

    estimates = StateSpaceData()
    ground_truth = StateSpaceData()
    ground_truth_state = kf.x

    # Generate control inputs, measurements, and update the Kalman filter
    for t in time:
        control = get_control_input(
            x=kf.x, desired=desired_state, gain_matrix=gain_matrix, limit=100
        )

        # Store the updated state for plotting
        estimates.append_step(t=t, x=kf.x, cov=kf.cov, u=control)
        ground_truth.append_step(t=t, x=ground_truth_state)

        # step through the filter
        kf.predict(u=control)
        ground_truth_state = ss.step(x=ground_truth_state, u=control)
        measurement = get_measurement(ss.C, state=ground_truth_state, noise=kf.R)
        kf.update(z=measurement)

    ss.plot_history(history=estimates)


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
    args = parser.parse_args()
    main(dir_path=args.dir)
