"""Basic docstring for my module."""

import argparse
from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from config.definitions import (
    DEFAULT_DISCRETIZATION,
    FIG_SIZE,
    MEASUREMENT_NOISE,
    PROCESS_NOISE,
)
from src.data_classes.pose import Pose2D
from src.data_classes.state_space import StateSpaceData, plot_history
from src.modules.controller import full_state_feedback, get_control_input
from src.modules.extended_kalman import ExtendedKalmanFilter
from src.modules.kalman import KalmanFilter
from src.modules.simulator import Simulator, mass_spring_damper_model, robot_model


class Pipeline(Enum):
    """Create an enumerator to choose which pipeline to run."""

    KF = auto()
    EKF = auto()
    EKF_SLAM = auto()
    CONTROLLER = auto()
    STATE_SPACE = auto()


def run_state_space_pipeline() -> None:
    """Run the main program with this function."""
    dt = DEFAULT_DISCRETIZATION
    ss_model = mass_spring_damper_model(discretization_dt=dt)
    ss_model.step_response(delta_t=dt, plot_response=True)
    ss_model.impulse_response(delta_t=dt, plot_response=True)


def run_kf_pipeline() -> None:
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
        process_noise=PROCESS_NOISE * np.eye(2),
        measurement_noise=MEASUREMENT_NOISE * np.eye(2),
        initial_x=np.array([[5.0], [5.0]]),
        initial_covariance=5 * np.eye(2),
    )

    sim = Simulator(
        state_space=ss,
        process_noise=kf.Q,
        measurement_noise=kf.R,
        initial_state=kf.x,
    )

    sim_history = StateSpaceData()

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


def run_ekf_pipeline():
    """Test the EKF algorithm."""
    logger.info("Running Extended Kalman Filter pipeline.")

    robot = robot_model()
    initial_pose = Pose2D(x=0.0, y=0.0, theta=np.pi / 2)
    ekf = ExtendedKalmanFilter(
        state_space_nonlinear=robot,
        process_noise=PROCESS_NOISE * np.eye(3),
        measurement_noise=MEASUREMENT_NOISE * np.eye(3),
        initial_x=initial_pose.as_vector(),
        initial_covariance=5 * np.eye(3),
    )
    plt.figure(figsize=FIG_SIZE)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.axis("equal")
    steps = 10
    for _i in range(2 * steps):
        vel = 1.0
        omega = 2 * np.pi / steps
        u = np.array([[vel], [omega]])
        ekf.predict(u=u)
        ekf.update(z=ekf.x, u=u)
        pose = Pose2D(
            x=ekf.x[0, 0],
            y=ekf.x[1, 0],
            theta=ekf.x[2, 0] % (2 * np.pi),
        )
        logger.info(f"x:\n{pose}")
        plt.arrow(
            x=pose.x,
            y=pose.y,
            dx=0.1 * np.cos(pose.theta),
            dy=0.1 * np.sin(pose.theta),
            width=0.01,
        )
        plt.draw()
    plt.show()
    plt.close()


def main(pipeline_id: str) -> None:
    """Process which pipeline to run."""
    if pipeline_id == Pipeline.KF.name:
        run_kf_pipeline()
    elif pipeline_id == Pipeline.EKF.name:
        run_ekf_pipeline()
    elif pipeline_id == Pipeline.STATE_SPACE.name:
        run_state_space_pipeline()
    else:
        msg = f"Invalid pipeline number: {pipeline_id}"
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Program complete.")


if __name__ == "__main__":  # pragma: no cover
    """Run the main program with this function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--pipeline",
        action="store",
        type=str,
        required=True,
        help="Choose which pipeline to run. (1, 2, 3, etc.)",
    )
    args = parser.parse_args()
    main(pipeline_id=args.pipeline)
