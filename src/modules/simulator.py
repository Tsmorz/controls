"""Basic docstring for my module."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from config.definitions import DEFAULT_DISCRETIZATION
from src.data_classes.map import Feature
from src.data_classes.pose import Pose2D
from src.modules.state_space import StateSpaceLinear, StateSpaceNonlinear


class KalmanSimulator:
    """Kalman filter implementation."""

    def __init__(
        self,
        state_space: StateSpaceLinear,
        process_noise: np.ndarray,
        measurement_noise: np.ndarray,
        initial_state: np.ndarray,
    ) -> None:
        """Initialize the Kalman Filter.

        :param state_space: linear state space model
        :param process_noise: Process noise covariance
        :param measurement_noise: Measurement noise covariance
        :param initial_state: Initial state estimate
        :return: None
        """
        self.state_space = state_space
        self.A: np.ndarray = state_space.A
        self.B: np.ndarray = state_space.B
        self.C: np.ndarray = state_space.C
        self.Q: np.ndarray = process_noise
        self.R: np.ndarray = measurement_noise
        self.x: np.ndarray = initial_state

    def step(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict the next state and error covariance.

        :param u: Control input
        """
        if u is None:
            u = np.zeros((self.B.shape[1], 1))
        scale = np.diag(self.Q)
        scale = np.reshape(scale, (self.Q.shape[0], 1))
        noise = np.random.normal(loc=0.0, scale=scale, size=(self.A.shape[0], 1))
        self.x = self.state_space.step(x=self.x, u=u) + noise
        return self.x

    def get_measurement(self) -> np.ndarray:
        """Get a measurement of the state.

        :return: Measurement of the state
        """
        scale = np.diag(self.R)
        scale = np.reshape(scale, (self.R.shape[0], 1))
        noise = np.random.normal(loc=0.0, scale=scale, size=(self.C.shape[0], 1))
        return self.C @ self.x + noise


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


def robot_model() -> StateSpaceNonlinear:
    """Create a StateSpaceNonlinear model of a wheeled robot."""

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

    motion_model = [
        pos_x_func,
        pos_y_func,
        heading_func,
    ]

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

    measurement_model = [
        measure_range_func,
        measure_angle_func,
    ]

    return StateSpaceNonlinear(
        motion_model=motion_model, measurement_model=measurement_model
    )


class SlamSimulator:
    """Kalman filter implementation."""

    def __init__(
        self,
        state_space_nl: StateSpaceNonlinear,
        process_noise: np.ndarray,
        measurement_noise: np.ndarray,
        initial_pose: Pose2D,
    ) -> None:
        """Initialize the Kalman Filter.

        :param state_space_nl: linear state space model
        :param process_noise: Process noise covariance
        :param measurement_noise: Measurement noise covariance
        :param initial_pose: Initial state estimate
        :return: None
        """
        self.state_space_nl = state_space_nl
        self.Q: np.ndarray = process_noise
        self.R: np.ndarray = measurement_noise
        self.pose: Pose2D = initial_pose
        self.history: list[tuple[Pose2D, Pose2D]] = []

    def step(self, u: np.ndarray) -> Pose2D:
        """Predict the next state and error covariance.

        :param u: Control input
        """
        scale = np.diag(self.Q)
        scale = np.reshape(scale, (self.Q.shape[0], 1))
        noise = np.random.normal(loc=0.0, scale=scale, size=(3, 1))
        x = self.state_space_nl.step(x=self.pose.as_vector(), u=u) + noise
        self.pose = Pose2D(x=x[0, 0], y=x[1, 0], theta=x[2, 0])
        return self.pose

    def get_measurement(self, feature: Feature) -> np.ndarray:
        """Calculate the range distance measurement between a pose and a feature.

        :param feature: Feature in the map
        :return: Range distance measurement
        """
        delta_x = feature.x - self.pose.x
        delta_y = feature.y - self.pose.y
        distance = np.sqrt(delta_x**2 + delta_y**2)

        angle = np.arctan2(delta_y, delta_x) - self.pose.theta

        return np.array([[distance], [angle]])

    def append_estimate(self, estimated_pose: Pose2D) -> None:
        """Update the state estimate based on an estimated pose."""
        self.history.append((estimated_pose, self.pose))

    def plot_results(self):
        plt.figure(figsize=(8, 8)).add_subplot(111)
        plt.axis("equal")
        plt.grid(True)
        plt.xlabel("x position")
        plt.ylabel("y position")
        plt.title("Robot Localization")
        for estimate_pose, true_pose in self.history:
            plt.plot(
                [true_pose.x, estimate_pose.x], [true_pose.y, estimate_pose.y], "k--"
            )
            plt.arrow(
                x=estimate_pose.x,
                y=estimate_pose.y,
                dx=0.1 * np.cos(estimate_pose.theta),
                dy=0.1 * np.sin(estimate_pose.theta),
                width=0.01,
                color="blue",
            )
            plt.arrow(
                x=true_pose.x,
                y=true_pose.y,
                dx=0.1 * np.cos(true_pose.theta),
                dy=0.1 * np.sin(true_pose.theta),
                width=0.01,
                color="red",
            )
            plt.draw()
            plt.pause(0.05)

        plt.show()
        plt.close()
