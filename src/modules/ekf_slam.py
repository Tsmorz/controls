"""Basic docstring for my module."""

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from src.data_classes.lie_algebra import SE3
from src.data_classes.map import Feature, make_random_map_planar
from src.data_classes.sensors import SensorType
from src.data_classes.slam import PoseMap
from src.modules.extended_kalman import ExtendedKalmanFilter
from src.modules.simulator import (
    SlamSimulator,
    add_measurement_to_plot,
    robot_model,
    state_to_se3,
)


def main() -> None:
    """Run the EKF for SLAM."""
    logger.info("Running Extended Kalman Filter pipeline.")

    robot = robot_model()
    pose = SE3(xyz=np.zeros((3,)), roll_pitch_yaw=np.zeros((3,)))
    pose_map = PoseMap(pose=pose)
    ekf = ExtendedKalmanFilter(
        state_space_nonlinear=robot,
        initial_x=pose_map.pose.as_vector(),
        initial_covariance=3 * np.eye(pose.as_vector().shape[0]),
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

                ekf.update(z=measurement.as_vector(), u=u, measurement_args=feature)
                if measurement.type == SensorType.DISTANCE_AND_BEARING:
                    add_measurement_to_plot(measurement, state=ekf.x)

        logger.info(f"state: {ekf.x.T}")
        pose = state_to_se3(state=ekf.x)
        sim.append_estimate(estimated_pose=pose, plot_pose=True)

    plt.show()
    plt.close()


if __name__ == "__main__":  # pragma: no cover
    """Test the EKF SLAM algorithm."""
    main()
