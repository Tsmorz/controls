"""Basic docstring for my module."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Patch
from matplotlib.pyplot import Axes, Figure

from config.definitions import (
    DEFAULT_DISCRETIZATION,
    DELTA_T,
    PAUSE_TIME,
    PLOT_ALPHA,
)
from src.data_classes.lie_algebra import SE3
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
        self.last_measurement: np.ndarray = np.array([])
        self.time_stamps: np.ndarray = np.arange(
            start=0.0, stop=20000 / DELTA_T, step=DELTA_T
        )
        self.controls = get_angular_velocities_for_box(
            steps=len(self.time_stamps), radius_steps=8
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
        measurement: np.ndarray,
        show_plot: bool = True,
    ) -> None:
        """Update the state estimate based on an estimated pose."""
        pose, cov = estimate
        self.history.append((pose, self.pose))

        old_poses = []
        for old_pose, _ in self.history[-20:]:
            old_poses.append(old_pose)
        plot_items: list[plt.Line2D | Patch] = []
        if show_plot:
            plot_items.append(self.pose.plot_se3(plot=self.sim_plot, color="red"))

            for ii, old_pose in enumerate(old_poses):
                plot_items.append(
                    old_pose.plot_se3(
                        plot=self.sim_plot, color="blue", alpha=ii / len(old_poses)
                    )
                )
            plot_items.append(self.plot_covariance(pose=pose, covariance=cov))
            plot_items.extend(self.plot_measurement(pose=pose, measurement=measurement))
            self.last_measurement = measurement
            plt.axis("equal")
            plt.pause(PAUSE_TIME)

            # remove the sensor measurements
            for item in plot_items:
                item.remove()

    def plot_covariance(self, pose: SE3, covariance: np.ndarray) -> Patch:
        """Add a drawing of the robot covariance to the plot."""
        xy_cov = np.linalg.eigvals(covariance[:2, :2])
        xy_cov = np.clip(xy_cov, a_min=-20, a_max=20)

        ellipse = Ellipse(
            xy=(float(pose.x), float(pose.y)),
            width=float(xy_cov[0]),
            height=float(xy_cov[1]),
            angle=np.rad2deg(np.arctan2(xy_cov[1], xy_cov[0])),
            fc="None",
            edgecolor="k",
            alpha=PLOT_ALPHA,
        )
        return self.sim_plot[1].add_patch(ellipse)

    def plot_measurement(
        self,
        measurement: np.ndarray,
        pose: SE3,
    ) -> list[plt.Line2D]:
        """Plot the simulation results."""
        # TODO: make the plot work for 3D features with azimuth and elevation
        rays: list[plt.Line2D] = []
        if not np.array_equal(self.last_measurement, measurement):
            fig, ax = self.sim_plot
            rays = []
            distance = measurement[0::3, 0]
            azimuth = measurement[1::3, 0]
            elevation = measurement[2::3, 0]
            dae = zip(distance, azimuth, elevation)
            for dist, azi, _ in dae:
                x1, x2 = pose.x, pose.x + dist * np.cos(pose.yaw + azi)
                y1, y2 = pose.y, pose.y + dist * np.sin(pose.yaw + azi)
                (m,) = ax.plot([x1, x2], [y1, y2], "k-", alpha=0.2)
                rays.append(m)
            return rays
        return rays
