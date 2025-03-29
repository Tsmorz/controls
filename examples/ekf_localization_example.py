"""Basic docstring for my module."""

import numpy as np
from loguru import logger

from config.definitions import (
    LOG_DECIMALS,
    MEASUREMENT_NOISE,
    SIGMA_OMEGA,
    SIGMA_VEL,
)
from ekf_slam_3d.data_classes.lie_algebra import SE3, state_to_se3
from ekf_slam_3d.data_classes.map import make_random_map_planar
from ekf_slam_3d.data_classes.sensors import Sensor, step_dynamics
from ekf_slam_3d.data_classes.slam import PoseMap
from ekf_slam_3d.modules.kalman_extended import ExtendedKalmanFilter
from ekf_slam_3d.modules.simulators import SlamSimulator
from ekf_slam_3d.modules.state_space import StateSpaceNonlinear


def pipeline(show_plot: bool) -> None:
    """Run the EKF for localization."""
    logger.info("Running EKF pipeline for localization.")
    robot_pose = SE3(xyz=np.zeros((3,)), roll_pitch_yaw=np.zeros((3,)))
    robot_pose.x = 20
    pose_map = PoseMap(pose=robot_pose)
    meas = np.array([])
    update_features = False

    process_noise = np.array([[SIGMA_VEL, 0], [0, SIGMA_OMEGA]])
    ekf = ExtendedKalmanFilter(
        state_space_nonlinear=StateSpaceNonlinear(motion_model=step_dynamics),
        initial_x=pose_map.pose.as_vector(),
        initial_covariance=0.1 * np.eye(robot_pose.as_vector().shape[0]),
        process_noise=process_noise,
        measurement_noise=MEASUREMENT_NOISE,
    )

    sim = SlamSimulator(
        state_space_nl=ekf.state_space_nonlinear,
        process_noise=ekf.Q,
        initial_pose=pose_map.pose,
        sim_map=make_random_map_planar(num_features=10, dim=(40, 40)),
    )

    i = 0
    for time in sim.time_stamps:
        control_input = np.array([[1.0], [2 * np.pi / 100]])
        sim.step(u=control_input)
        ekf.predict(u=control_input)
        pose_map.pose = SE3(xyz=ekf.x[0:3, 0], roll_pitch_yaw=ekf.x[3:6, 0])

        # update the state estimate with the measurements
        frac, whole = np.modf(time)
        if whole % 3 == 0 and time > 0.0:
            # TODO - update the heading with the magnetometer
            ekf.x[5, 0] = sim.pose.yaw + np.random.normal(0, 0.1)

        if whole % 15 == 0 and time > 0.0:
            meas = Sensor.DIST_AZI_ELE.func(
                state=sim.pose.as_vector(), features=sim.map.features
            )
            ekf.update(
                z=meas,
                sensor=Sensor.DIST_AZI_ELE.func,
                u=control_input,
                measurement_args=sim.map.features,
            )

        # save new features to the map
        if update_features:
            for feat in sim.map.features:
                pose_map.map.append_feature(feature=feat)

        sim.append_result(
            estimate=(state_to_se3(state=ekf.x), ekf.cov),
            measurement=meas,
            show_plot=show_plot,
        )

        if show_plot:
            x_str = np.array2string(ekf.x.T, precision=LOG_DECIMALS, floatmode="fixed")
            logger.info(f"state: {x_str}")
        i += 1


if __name__ == "__main__":
    pipeline(show_plot=True)
