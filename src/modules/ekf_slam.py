"""Basic docstring for my module."""

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from src.data_classes.lie_algebra import SE2
from src.data_classes.map import Feature, make_random_map_planar
from src.data_classes.sensors import SensorType
from src.data_classes.slam import PoseMap
from src.modules.extended_kalman import ExtendedKalmanFilter
from src.modules.simulator import SlamSimulator
from src.modules.state_space import StateSpaceNonlinear


def pos_x_func(state_control: np.ndarray) -> np.ndarray:
    """Find the x position given the state and control vectors."""
    pose = SE2(x=state_control[0, 0], y=state_control[1, 0], theta=state_control[2, 0])
    vel, omega = state_control[-2:, 0]
    return vel * np.cos(pose.theta) + pose.x


def pos_y_func(state_control: np.ndarray) -> np.ndarray:
    """Find the y position given the state and control vectors."""
    pose = SE2(x=state_control[0, 0], y=state_control[1, 0], theta=state_control[2, 0])
    vel, omega = state_control[-2:, 0]
    return vel * np.sin(pose.theta) + pose.y


def heading_func(state_control: np.ndarray) -> np.ndarray:
    """Find the heading given the state and control vectors."""
    pose = SE2(x=state_control[0, 0], y=state_control[1, 0], theta=state_control[2, 0])
    vel, omega = state_control[-2:, 0]
    return np.array(pose.theta + omega)


def measure_range_func(state_control: np.ndarray, feature: Feature) -> np.ndarray:
    """Find the x position given the state and control vectors."""
    pose = SE2(x=state_control[0, 0], y=state_control[1, 0], theta=state_control[2, 0])
    delta_x = feature.x - pose.x
    delta_y = feature.y - pose.y
    distance = np.sqrt(delta_x**2 + delta_y**2)
    return distance


def measure_angle_func(state_control: np.ndarray, feature: Feature) -> np.ndarray:
    """Find the y position given the state and control vectors."""
    pose = SE2(x=state_control[0, 0], y=state_control[1, 0], theta=state_control[2, 0])
    delta_x = feature.x - pose.x
    delta_y = feature.y - pose.y
    angle = np.arctan2(delta_y, delta_x) - pose.theta
    return angle


def robot_model() -> StateSpaceNonlinear:
    """Create a StateSpaceNonlinear model of a wheeled robot."""
    motion_model = [
        pos_x_func,
        pos_y_func,
        heading_func,
    ]

    measurement_model = [
        measure_range_func,
        measure_angle_func,
    ]

    return StateSpaceNonlinear(
        motion_model=motion_model, measurement_model=measurement_model
    )


def state_to_se2(state: np.ndarray) -> SE2:
    """Map the state vector to SE2."""
    return SE2(x=state[0, 0], y=state[1, 0], theta=state[2, 0])


def main() -> None:
    """Run the EKF for SLAM."""
    logger.info("Running Extended Kalman Filter pipeline.")

    robot = robot_model()
    pose_map = PoseMap(pose=SE2(x=0.0, y=0.0, theta=0.0))
    ekf = ExtendedKalmanFilter(
        state_space_nonlinear=robot,
        initial_x=pose_map.pose.as_vector(),
        initial_covariance=3 * np.eye(3),
    )

    sim = SlamSimulator(
        state_space_nl=robot,
        process_noise=ekf.Q,
        measurement_noise=ekf.R,
        initial_pose=pose_map.pose,
        steps=100,
    )

    true_map = make_random_map_planar(num_features=3, dim=(15, 15))

    linear_vel = 0.5
    for time, angular_vel in enumerate(sim.controls):
        u = np.array([[linear_vel], [angular_vel]])
        sim.step(u=u)
        ekf.predict(u=u)
        pose_map.from_vector(ekf_state=ekf.x)

        if (time / 20) % 1 == 0 and time != 0:
            for ii, feature in enumerate(true_map.features):
                feature_new = Feature(x=time, y=time, id=ii * time)
                measurement = sim.get_measurement(
                    feature=feature,
                    sensor_type=SensorType.DISTANCE_AND_BEARING,
                )
                logger.info(f"Measurement={measurement}")
                pose_map.map.append_feature(feature=feature_new)

                logger.info(ekf.x.T)
                ekf.update(z=measurement.as_vector(), u=u, measurement_args=feature)
                if measurement.type == SensorType.DISTANCE_AND_BEARING:
                    sim.add_measurement_to_plot(measurement, state=ekf.x)

        pose = state_to_se2(state=ekf.x)
        sim.append_estimate(estimated_pose=pose, plot_pose=True)

    plt.show()
    plt.close()


if __name__ == "__main__":  # pragma: no cover
    """Test the EKF SLAM algorithm."""
    main()
