"""Basic docstring for my module."""

import numpy as np
from loguru import logger

from config.definitions import (
    LOG_DECIMALS,
    MEASUREMENT_NOISE,
    SIGMA_OMEGA,
    SIGMA_VEL,
)
from src.data_classes.lie_algebra import SE3, state_to_se3
from src.data_classes.map import make_random_map_planar
from src.data_classes.sensors import DistanceAzimuthElevation, Dynamics
from src.data_classes.slam import PoseMap
from src.modules.kalman_extended import ExtendedKalmanFilter
from src.modules.simulators import SlamSimulator
from src.modules.state_space import StateSpaceNonlinear


def pipeline() -> None:
    """Run the EKF for SLAM."""
    logger.info("Running Extended Kalman Filter pipeline.")
    robot_pose = SE3(xyz=np.zeros((3,)), roll_pitch_yaw=np.zeros((3,)))
    pose_map = PoseMap(pose=robot_pose)
    meas = None

    process_noise = np.array([[SIGMA_VEL, 0], [0, SIGMA_OMEGA]])
    ekf = ExtendedKalmanFilter(
        state_space_nonlinear=StateSpaceNonlinear(motion_model=Dynamics),
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
        control_input = np.array([[1.5], [sim.controls[i]]])
        sim.step(u=control_input)
        ekf.predict(u=control_input)
        pose_map.pose = SE3(xyz=ekf.x[0:3, 0], roll_pitch_yaw=ekf.x[3:6, 0])

        frac, whole = np.modf(time)
        if whole % 5 == 0 and time > 0.0:
            meas = DistanceAzimuthElevation(
                state=sim.pose.as_vector(),
                features=sim.map.features,
                noise=MEASUREMENT_NOISE,
            )
            ekf.update(z=meas, u=control_input, measurement_args=sim.map.features)

            # save new features to the map
            for feat in sim.map.features:
                pose_map.map.append_feature(feature=feat)

        sim.append_result(
            estimate=(state_to_se3(state=ekf.x), ekf.cov),
            measurement=meas,
        )

        x_str = np.array2string(ekf.x.T, precision=LOG_DECIMALS, floatmode="fixed")
        logger.info(f"state: {x_str}")
        cov_str = np.array2string(
            np.diag(ekf.cov), precision=LOG_DECIMALS, floatmode="fixed"
        )
        logger.info(f"cov: {cov_str}")
        i += 1


if __name__ == "__main__":
    pipeline()
