"""Basic docstring for my module."""

import argparse
from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from config.definitions import (
    DEFAULT_DISCRETIZATION,
)
from src.data_classes.map import Map
from src.data_classes.pose import SE2
from src.data_classes.sensors import SensorType
from src.data_classes.state_space import StateSpaceData, plot_history
from src.modules.controller import full_state_feedback, get_control_input
from src.modules.extended_kalman import ExtendedKalmanFilter
from src.modules.kalman import KalmanFilter
from src.modules.simulator import (
    KalmanSimulator,
    SlamSimulator,
    get_angular_velocities_for_box,
    mass_spring_damper_model,
    robot_model,
)


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
        initial_x=np.array([[5.0], [5.0]]),
        initial_covariance=5 * np.eye(2),
    )

    sim = KalmanSimulator(
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


def run_ekf_pipeline() -> None:
    """Test the EKF algorithm."""

    def state_to_se2(state: np.ndarray) -> SE2:
        """Map the state vector to SE2."""
        return SE2(x=state[0, 0], y=state[1, 0], theta=state[2, 0])

    logger.info("Running Extended Kalman Filter pipeline.")

    robot = robot_model()
    initial_pose = SE2(x=-1.0, y=0.0, theta=0.0)
    ekf = ExtendedKalmanFilter(
        state_space_nonlinear=robot,
        initial_x=initial_pose.as_vector(),
        initial_covariance=3 * np.eye(3),
    )

    sim = SlamSimulator(
        state_space_nl=robot,
        process_noise=ekf.Q,
        measurement_noise=ekf.R,
        initial_pose=initial_pose,
    )

    sim_map = Map()
    sim_map.make_random_map_planar(num_features=3, dim=(15, 15))

    plt.figure(figsize=(8, 8)).add_subplot(111)
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.title("Robot Localization")

    omegas = get_angular_velocities_for_box(steps=100, radius_steps=6)
    for ii, omega in enumerate(omegas):
        logger.info(f"time={ii}s")
        u = np.array([[0.5], [omega]])
        sim.step(u=u)
        ekf.predict(u=u)

        if (ii / 20) % 1 == 0 and ii != 0:
            for feature in sim_map.features:
                sensor = SensorType.DISTANCE_AND_BEARING
                measurement = sim.get_measurement(feature=feature, sensor_type=sensor)
                logger.info(f"Measurement={measurement}")

                ekf.update(z=measurement.as_vector(), u=u, measurement_args=feature)

                if measurement.type == SensorType.DISTANCE_AND_BEARING:
                    m = measurement.as_vector()
                    pose = state_to_se2(state=ekf.x)
                    x1, x2 = pose.x, pose.x + m[0, 0] * np.cos(pose.theta + m[1, 0])
                    y1, y2 = pose.y, pose.y + m[0, 0] * np.sin(pose.theta + m[1, 0])
                    plt.plot([x1, x2], [y1, y2], "k-", alpha=0.2)

        pose = state_to_se2(state=ekf.x)
        sim.append_estimate(state_to_se2(state=ekf.x))

        plt.plot([sim.pose.x, pose.x], [sim.pose.y, pose.y], "k-", alpha=0.8)
        plt.arrow(
            x=pose.x,
            y=pose.y,
            dx=0.1 * np.cos(pose.theta),
            dy=0.1 * np.sin(pose.theta),
            width=0.01,
            color="blue",
        )
        plt.arrow(
            x=sim.pose.x,
            y=sim.pose.y,
            dx=0.1 * np.cos(sim.pose.theta),
            dy=0.1 * np.sin(sim.pose.theta),
            width=0.01,
            color="red",
        )
        plt.draw()
        plt.pause(0.05)

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
