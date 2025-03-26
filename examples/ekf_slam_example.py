"""Basic docstring for my module."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.patches import Ellipse
from matplotlib.pyplot import Axes, Figure

from config.definitions import (
    LOG_DECIMALS,
    MEASUREMENT_NOISE,
    PAUSE_TIME,
    SIGMA_OMEGA,
    SIGMA_VEL,
    VECTOR_LENGTH,
)
from src.data_classes.lie_algebra import SE3, state_to_se3
from src.data_classes.map import make_random_map_planar
from src.data_classes.sensors import DistanceAzimuthElevation, Dynamics
from src.data_classes.slam import Map, PoseMap
from src.modules.kalman_extended import ExtendedKalmanFilter
from src.modules.state_space import StateSpaceNonlinear


def get_angular_velocities_for_box(steps: int, radius_steps: int) -> list[float]:
    """Create the angular velocity control inputs for a box."""
    side_length = int(steps / 4)
    one_side = side_length * [0] + radius_steps * [np.pi / 2 / radius_steps]
    turning_rates = one_side + one_side + one_side + one_side
    return turning_rates


class SlamSimulator:
    """Kalman filter implementation."""

    def __init__(
        self,
        state_space_nl: StateSpaceNonlinear,
        process_noise: np.ndarray,
        initial_pose: SE3,
        sim_map: Map,
        steps: int = 100,
    ) -> None:
        """Initialize the Kalman Filter.

        :param state_space_nl: linear state space model
        :param process_noise: Process noise covariance
        :param initial_pose: Initial state estimate
        :param steps: Number of simulation steps
        :param sim_map: Optional map for visualization
        :return: None
        """
        self.state_space_nl = state_space_nl
        self.Q: np.ndarray = process_noise
        self.pose: SE3 = initial_pose
        self.history: list[tuple[SE3, SE3]] = []
        self.controls = get_angular_velocities_for_box(steps=steps, radius_steps=6)
        self.map: Map = sim_map
        self.last_measurement: Optional[DistanceAzimuthElevation] = None
        self.steps: int = steps

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        plt.axis("equal")
        plt.grid(True)
        plt.xlabel("x position")
        plt.ylabel("y position")
        plt.title("Robot Localization")
        for feature in self.map.features:
            ax.plot(feature.x, feature.y, "k*")
        self.sim_plot: tuple[Figure, Axes] = (fig, ax)

    def step(self, u: np.ndarray) -> SE3:
        """Predict the next state and error covariance.

        :param u: Control input
        """
        scale = np.reshape(np.diag(self.Q), (self.Q.shape[0], 1))
        noise = np.random.normal(loc=0.0, scale=scale, size=(self.Q.shape[0], 1))
        x = self.state_space_nl.step(x=self.pose.as_vector(), u=u + noise)
        self.pose = SE3(xyz=x[0:3], roll_pitch_yaw=x[3:6])
        return self.pose

    def append_result(
        self,
        estimate: tuple[SE3, np.ndarray],
        measurement: Optional[DistanceAzimuthElevation] = None,
    ) -> None:
        """Update the state estimate based on an estimated pose."""
        pose, cov = estimate
        self.history.append((pose, self.pose))
        self.plot_pose(pose=pose, color="blue")
        self.plot_pose(pose=self.pose, color="red")
        if self.last_measurement != measurement and measurement is not None:
            self.plot_measurement(pose=pose, measurement=measurement)
            self.last_measurement = measurement

        cov_plot = self.plot_covariance(pose=pose, covariance=cov)
        plt.axis("equal")
        plt.draw()
        plt.pause(PAUSE_TIME)
        cov_plot.remove()

    def plot_pose(self, pose: SE3, color: str) -> None:
        """Add a drawing of the robot pose to the plot."""
        fig, ax = self.sim_plot
        ax.plot([pose.x, pose.x], [pose.y, pose.y], "k-", alpha=0.8)
        dx, dy = VECTOR_LENGTH * np.cos(pose.yaw), VECTOR_LENGTH * np.sin(pose.yaw)
        ax.arrow(x=pose.x, y=pose.y, dx=dx, dy=dy, width=0.01, color=color)
        plt.draw()

    def plot_covariance(self, pose: SE3, covariance: np.ndarray):
        """Add a drawing of the robot covariance to the plot."""
        x_cov, y_cov = np.linalg.eigvals(covariance[:2, :2])
        ang_cov = np.rad2deg(np.arctan2(y_cov, x_cov))
        ellipse = Ellipse(
            xy=(float(pose.x), float(pose.y)),
            width=x_cov,
            height=y_cov,
            angle=ang_cov,
            fc="None",
            edgecolor="k",
        )
        cov_ellipse = self.sim_plot[1].add_patch(ellipse)
        return cov_ellipse

    def plot_measurement(
        self,
        measurement: DistanceAzimuthElevation,
        pose: SE3,
    ) -> None:
        """Plot the simulation results."""
        # TODO: make the plot work for 3D features with azimuth and elevation
        fig, ax = self.sim_plot
        rays = []
        for dist, azi, _ in zip(
            measurement.distance, measurement.azimuth, measurement.elevation
        ):
            x1, x2 = pose.x, pose.x + dist * np.cos(pose.yaw + azi)
            y1, y2 = pose.y, pose.y + dist * np.sin(pose.yaw + azi)
            (m,) = plt.plot([x1, x2], [y1, y2], "k-", alpha=0.2)
            rays.append(m)
        ax.axis("equal")
        plt.pause(PAUSE_TIME)

        # remove the sensor measurements
        for ray in rays:
            ray.remove()


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
        sim_map=make_random_map_planar(num_features=3, dim=(20, 20)),
        steps=200,
    )

    for time in range(sim.steps):
        linear_vel, angular_vel = 0.5, sim.controls[time]
        control_input = np.array([[linear_vel], [angular_vel]])
        sim.step(u=control_input)
        ekf.predict(u=control_input)
        pose_map.pose = SE3(xyz=ekf.x[0:3, 0], roll_pitch_yaw=ekf.x[3:6, 0])

        if (time / 15) % 1 == 0 and time != 0:
            meas = DistanceAzimuthElevation(
                state=sim.pose.as_vector(),
                features=sim.map.features,
                noise=None,
            )
            ekf.update(z=meas, u=control_input, measurement_args=sim.map.features)

            for feat in sim.map.features:
                pose_map.map.append_feature(feature=feat)

        robot_pose = state_to_se3(state=ekf.x)
        sim.append_result(estimate=(robot_pose, ekf.cov), measurement=meas)

        x_str = np.array2string(ekf.x.T, precision=LOG_DECIMALS, floatmode="fixed")
        logger.info(f"state: {x_str}")
        cov_str = np.array2string(
            np.diag(ekf.cov), precision=LOG_DECIMALS, floatmode="fixed"
        )
        logger.info(f"cov: {cov_str}")


if __name__ == "__main__":
    pipeline()
