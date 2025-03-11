"""Basic docstring for my module."""

import argparse

import numpy as np
from loguru import logger
from scipy.signal import place_poles

from config.definitions import MEASUREMENT_NOISE, PROCESS_NOISE
from src.modules.filter import KalmanFilter, get_measurement
from src.modules.state_space import StateSpace, StateSpaceData, mass_spring_damper_model


def main(dir_path: str) -> None:
    """Pipeline to run the repo code."""
    logger.info(f"Path to directory: {dir_path}")
    dt = 0.05
    time = np.arange(0, 10, dt).tolist()
    ss = mass_spring_damper_model(discretization_dt=dt)
    desired_eigenvalues = np.array([0.89 + 0.29j, 0.89 - 0.29j])

    kf = KalmanFilter(
        state_space=ss,
        process_noise=PROCESS_NOISE * np.eye(2),
        measurement_noise=MEASUREMENT_NOISE * np.eye(2),
        initial_state=np.array([[2.0], [2.0]]),
        initial_covariance=np.eye(2),
    )

    # Generate random control inputs and measurements and update the Kalman filter
    results = StateSpaceData()
    ground_truth_state = kf.x
    gain_matrix = full_state_feedback(ss, desired_eigenvalues)
    desired = np.array([[0], [0]])
    for t in time:
        error = kf.x - desired
        control = -gain_matrix @ error
        clip_limit = 10000
        control = np.clip(control, -clip_limit, clip_limit)

        # Store the updated state for plotting
        results.append_step(t=t, x=kf.x, cov=kf.cov, u=control)

        # step through the filter
        kf.predict(u=control)
        ground_truth_state = ss.step(x=ground_truth_state, u=control)
        measurement = get_measurement(ss.C, state=ground_truth_state, noise=kf.R)
        kf.update(z=measurement)

    ss.plot_history(history=results)


def full_state_feedback(state_space: StateSpace, desired_eigenvalues: np.ndarray):
    """Calculate the feedback gains for a desired response.

    :param state_space: State-space model
    :param desired_eigenvalues: Desired eigenvalues
    :return: Feedback gains
    """
    place_result = place_poles(state_space.A, state_space.B, desired_eigenvalues)
    K = place_result.gain_matrix

    augmented = state_space.A - state_space.B @ K
    if np.linalg.eigvals(augmented).all() != desired_eigenvalues.all():
        msg = "The desired eigenvalues are not correct."
        logger.error(msg)
        raise ValueError(msg)
    return K


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
