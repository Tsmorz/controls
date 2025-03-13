"""Basic docstring for my module."""

import argparse

import numpy as np
from loguru import logger

from config.definitions import MEASUREMENT_NOISE, PROCESS_NOISE
from src.modules.controller import full_state_feedback, get_control_input
from src.modules.kalman import KalmanFilter
from src.modules.simulator import Simulator
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

    sim = Simulator(
        state_space=ss,
        process_noise=kf.Q,
        measurement_noise=kf.R,
        initial_state=kf.x,
    )

    estimates = StateSpaceData()
    ground_truth = StateSpaceData()

    # Generate control inputs, measurements, and update the Kalman filter
    for t in time:
        control = get_control_input(
            x=kf.x, desired=desired_state, gain_matrix=gain_matrix, limit=100
        )

        # Store the updated state for plotting
        estimates.append_step(t=t, x=kf.x, cov=kf.cov, u=control)
        ground_truth.append_step(t=t, x=sim.x)

        # Simulate the system
        sim.step(u=control)
        measurement = sim.get_measurement()

        # Step through the filter
        kf.predict(u=control)
        kf.update(z=measurement)

    ss.plot_history(history=estimates)
    ss.plot_states(history=estimates)


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
