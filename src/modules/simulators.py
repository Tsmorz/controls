"""Basic docstring for my module."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.pyplot import Axes, Figure

from config.definitions import (
    DEFAULT_DISCRETIZATION,
    DELTA_T,
    PAUSE_TIME,
    VECTOR_LENGTH,
)
from src.data_classes.lie_algebra import SE3
from src.data_classes.sensors import DistanceAzimuthElevation
from src.data_classes.slam import Map
from src.modules.controller import get_angular_velocities_for_box
from src.modules.state_space import StateSpaceLinear, StateSpaceNonlinear


def mass_spring_damper_model(
    mass: float = 0.5,
    spring_const: float = 20.0,
    damping: float = 0.4,
    discretization_dt: float = DEFAULT_DISCRETIZATION,
) -> StateSpaceLinear:  # pragma: no cover
    """Calculate a simple mass spring damper model.

    :param mass: Mass of the system
    :param spring_const: Spring constant
    :param damping: Damping coefficient
    :param discretization_dt: Desired discrete time step size
    :return: state-space model
    """
    model = StateSpaceLinear(
        A=np.array([[0.0, 1.0], [-spring_const / mass, -damping / mass]]),
        B=np.array([[0.0], [1.0 / mass]]),
    )
    model.continuous_to_discrete(discretization_dt)
    return model


class SlamSimulator:
    """Kalman filter implementation."""

    def __init__(
        self,
        state_space_nl: StateSpaceNonlinear,
        process_noise: np.ndarray,
        initial_pose: SE3,
        sim_map: Map,
    ) -> None:
        """Initialize the Kalman Filter.

        :param state_space_nl: linear state space model
        :param process_noise: Process noise covariance
        :param initial_pose: Initial state estimate
        :param sim_map: Optional map for visualization
        :return: None
        """
        self.state_space_nl = state_space_nl
        self.Q: np.ndarray = process_noise
        self.pose: SE3 = initial_pose
        self.history: list[tuple[SE3, SE3]] = []
        self.map: Map = sim_map
        self.last_measurement: Optional[DistanceAzimuthElevation] = None
        self.time_stamps: np.ndarray = np.arange(
            start=0.0, stop=100 / DELTA_T, step=DELTA_T
        )
        self.controls = get_angular_velocities_for_box(
            steps=len(self.time_stamps), radius_steps=6
        )

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
