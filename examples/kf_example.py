"""Basic docstring for my module."""

import numpy as np
from loguru import logger

from config.definitions import (
    DEFAULT_DISCRETIZATION,
)
from src.data_classes.state_history import StateHistory, plot_history
from src.modules.controller import full_state_feedback, get_control_input
from src.modules.kalman import KalmanFilter
from src.modules.simulator import (
    KalmanSimulator,
    mass_spring_damper_model,
)


def pipeline() -> None:
    """Pipeline to run the repo code."""
    logger.info("Running Kalman Filter pipeline...")

    dt = DEFAULT_DISCRETIZATION
    time = np.arange(0, 10, dt).tolist()
    ss = mass_spring_damper_model(discretization_dt=dt)

    # find the desired control gains
    desired_eigenvalues = np.array([0.89 + 0.29j, 0.89 - 0.29j])
    gain_matrix = full_state_feedback(ss, desired_eigenvalues)
    desired_state = np.array([[0], [0]])

    # initialize the kalman filter
    kf = KalmanFilter(
        state_space=ss,
        initial_x=np.array([[5.0], [5.0]]),
        initial_covariance=5 * np.eye(2),
    )

    sim = KalmanSimulator(
        state_space=ss,
        process_noise=kf.Q,
        measurement_noise=kf.R,
        initial_state=kf.x,
    )

    sim_history = StateHistory()

    # Generate control inputs, measurements, and update the Kalman filter
    for t in time:
        u = get_control_input(x=kf.x, desired=desired_state, gain_matrix=gain_matrix)

        # Store the updated state for plotting
        sim_history.append_step(t=t, x=kf.x, cov=kf.cov, u=u, x_truth=sim.x)

        # Simulate the system
        sim.step(u=u)
        measurement = sim.get_measurement()

        # Step through the filter
        kf.predict(u=u)
        kf.update(z=measurement)

    plot_history(history=sim_history)


if __name__ == "__main__":
    pipeline()
