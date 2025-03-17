"""Basic docstring for my module."""

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from src.data_classes.map import Feature, Map
from src.data_classes.pose import SE2
from src.data_classes.sensors import SensorType
from src.modules.extended_kalman import ExtendedKalmanFilter
from src.modules.simulator import SlamSimulator
from src.modules.state_space import StateSpaceNonlinear


def pos_x_func(state_control: np.ndarray) -> np.ndarray:
    """Find the x position given the state and control vectors."""
    pos_x, _, theta, vel, _ = state_control
    return vel * np.cos(theta) + pos_x


def pos_y_func(state_control: np.ndarray) -> np.ndarray:
    """Find the y position given the state and control vectors."""
    _, pos_y, theta, vel, _ = state_control
    return vel * np.sin(theta) + pos_y


def heading_func(state_control: np.ndarray) -> np.ndarray:
    """Find the heading given the state and control vectors."""
    _, _, theta, vel, theta_dot = state_control
    return np.array(theta + theta_dot)


def measure_range_func(state_control: np.ndarray, feature: Feature) -> np.ndarray:
    """Find the x position given the state and control vectors."""
    pos_x, pos_y, _, _, _ = state_control
    delta_x = feature.x - pos_x
    delta_y = feature.y - pos_y
    distance = np.sqrt(delta_x**2 + delta_y**2)
    return distance


def measure_angle_func(state_control: np.ndarray, feature: Feature) -> np.ndarray:
    """Find the y position given the state and control vectors."""
    pos_x, pos_y, theta, _, _ = state_control
    delta_x = feature.x - pos_x
    delta_y = feature.y - pos_y
    angle = np.arctan2(delta_y, delta_x) - theta
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
        steps=100,
    )

    sim_map = Map()
    sim_map.make_random_map_planar(num_features=3, dim=(15, 15))

    states = {"x": initial_pose.x, "y": initial_pose.y, "theta": initial_pose.theta}
    for time, omega in enumerate(sim.controls):
        u = np.array([[0.5], [omega]])
        sim.step(u=u)
        ekf.predict(u=u)

        if (time / 20) % 1 == 0 and time != 0:
            for feature in sim_map.features:
                feature_new = Feature(x=time, y=time, id=time * 10)
                measurement = sim.get_measurement(
                    feature=feature,
                    sensor_type=SensorType.DISTANCE_AND_BEARING,
                )
                logger.info(f"Measurement={measurement}")

                if feature_new.id not in [feature.id for feature in sim_map.features]:
                    states["x"] = ekf.x[0, 0]
                    states["y"] = ekf.x[1, 0]
                    states["theta"] = ekf.x[2, 0]
                    states[f"feature_{time}_x"] = feature_new.x
                    states[f"feature_{time}_y"] = feature_new.y
                    dict_to_array = np.array(list(states.values()))
                    dict_to_array = np.reshape(dict_to_array, (len(dict_to_array), 1))
                    logger.info(states)
                    array_str = np.array2string(
                        dict_to_array, precision=2, floatmode="fixed"
                    )
                    logger.info(array_str)

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
