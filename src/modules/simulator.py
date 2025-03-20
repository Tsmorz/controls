"""Basic docstring for my module."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from config.definitions import DEFAULT_DISCRETIZATION
from src.data_classes.lie_algebra import SE3
from src.data_classes.map import Feature
from src.data_classes.sensors import Bearing, Distance, DistanceAndBearing, SensorType
from src.modules.state_space import StateSpaceLinear, StateSpaceNonlinear


def state_to_se3(state: np.ndarray) -> SE3:
    """Map the state vector to SE2."""
    return SE3(xyz=state[0:3, 0], roll_pitch_yaw=state[3:6, 0])


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


def get_angular_velocities_for_box(steps: int, radius_steps: int) -> list[float]:
    """Create the angular velocity control inputs for a box."""
    side_length = int(steps / 4)
    one_side = side_length * [0] + radius_steps * [np.pi / 2 / radius_steps]
    turning_rates = one_side + one_side + one_side + one_side
    return turning_rates


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


def pos_x_func(state_control: np.ndarray) -> np.ndarray:
    """Find the x position given the state and control vectors."""
    pos_x, pos_y, pos_z, roll, pitch, yaw, vel, omega = state_control
    return vel * np.cos(yaw) + pos_x


def pos_y_func(state_control: np.ndarray) -> np.ndarray:
    """Find the y position given the state and control vectors."""
    pos_x, pos_y, pos_z, roll, pitch, yaw, vel, omega = state_control
    return vel * np.sin(yaw) + pos_y


def pos_z_func(state_control: np.ndarray) -> np.ndarray:
    """Find the y position given the state and control vectors."""
    pos_x, pos_y, pos_z, roll, pitch, yaw, vel, omega = state_control
    return pos_z


def roll_func(state_control: np.ndarray) -> np.ndarray:
    """Find the heading given the state and control vectors."""
    pos_x, pos_y, pos_z, roll, pitch, yaw, vel, omega = state_control
    return roll


def pitch_func(state_control: np.ndarray) -> np.ndarray:
    """Find the heading given the state and control vectors."""
    pos_x, pos_y, pos_z, roll, pitch, yaw, vel, omega = state_control
    return pitch


def yaw_func(state_control: np.ndarray) -> np.ndarray:
    """Find the heading given the state and control vectors."""
    pos_x, pos_y, pos_z, roll, pitch, yaw, vel, omega = state_control
    return np.array(yaw + omega)


def measure_range_func(state_control: np.ndarray, feature: Feature) -> np.ndarray:
    """Find the x position given the state and control vectors."""
    pos_x, pos_y, pos_z, roll, pitch, yaw, vel, omega = state_control
    delta_x = feature.x - pos_x
    delta_y = feature.y - pos_y
    delta_z = feature.z - pos_z
    distance = np.sqrt(delta_x**2 + delta_y**2 + delta_z)
    return distance


def measure_angle_func(state_control: np.ndarray, feature: Feature) -> np.ndarray:
    """Find the y position given the state and control vectors."""
    pos_x, pos_y, pos_z, roll, pitch, yaw, vel, omega = state_control
    delta_x = feature.x - pos_x
    delta_y = feature.y - pos_y
    angle = np.arctan2(delta_y, delta_x) - yaw
    return angle


def robot_model() -> StateSpaceNonlinear:
    """Create a StateSpaceNonlinear model of a wheeled robot."""
    motion_model = [
        pos_x_func,
        pos_y_func,
        pos_z_func,
        roll_func,
        pitch_func,
        yaw_func,
    ]

    measurement_model = [
        measure_range_func,
        measure_angle_func,
    ]

    return StateSpaceNonlinear(
        motion_model=motion_model, measurement_model=measurement_model
    )


def add_measurement_to_plot(measurement, state: np.ndarray) -> None:
    """Plot the simulation results."""
    m = measurement.as_vector()
    pose = state_to_se3(state=state)
    x1, x2 = pose.x, pose.x + m[0, 0] * np.cos(pose.yaw + m[1, 0])
    y1, y2 = pose.y, pose.y + m[0, 0] * np.sin(pose.yaw + m[1, 0])
    plt.plot([x1, x2], [y1, y2], "k-", alpha=0.2)


class SlamSimulator:
    """Kalman filter implementation."""

    def __init__(
        self,
        state_space_nl: StateSpaceNonlinear,
        process_noise: np.ndarray,
        measurement_noise: np.ndarray,
        initial_pose: SE3,
        steps: int = 100,
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
        self.pose: SE3 = initial_pose
        self.history: list[tuple[SE3, SE3]] = []
        self.controls = get_angular_velocities_for_box(steps=steps, radius_steps=6)

        plt.figure(figsize=(8, 8)).add_subplot(111)
        plt.axis("equal")
        plt.grid(True)
        plt.xlabel("x position")
        plt.ylabel("y position")
        plt.title("Robot Localization")
        self.figure: plt.Figure

    def step(self, u: np.ndarray) -> SE3:
        """Predict the next state and error covariance.

        :param u: Control input
        """
        scale = np.diag(self.Q)
        scale = np.reshape(scale, (self.Q.shape[0], 1))
        noise = np.random.normal(loc=0.0, scale=scale, size=(self.Q.shape[0], 1))
        x = self.state_space_nl.step(x=self.pose.as_vector(), u=u) + noise
        self.pose = SE3(xyz=x[0:3], roll_pitch_yaw=x[3:6])
        return self.pose

    def get_measurement(
        self, feature: Feature, sensor_type: SensorType
    ) -> DistanceAndBearing | Distance | Bearing:
        """Calculate the range distance measurement between a pose and a feature.

        :param feature: Feature in the map
        :param sensor_type: Sensor used for measurement
        :return: Range distance measurement
        """
        if sensor_type == SensorType.DISTANCE:
            return Distance(ground_truth=self.pose, features=[feature])
        elif sensor_type == SensorType.BEARING:
            return Bearing(ground_truth=self.pose, features=[feature])
        elif sensor_type == SensorType.DISTANCE_AND_BEARING:
            return DistanceAndBearing(ground_truth=self.pose, features=[feature])
        else:
            msg = f"Unsupported sensor type: {sensor_type}"
            logger.error(msg)
            raise ValueError(msg)

    def append_estimate(self, estimated_pose: SE3, plot_pose: bool) -> None:
        """Update the state estimate based on an estimated pose."""
        self.history.append((estimated_pose, self.pose))

        if plot_pose:
            plt.plot(
                [float(self.pose.x), float(estimated_pose.x)],
                [float(self.pose.y), float(estimated_pose.y)],
                "k-",
                alpha=0.8,
            )
            self.plot_pose(pose=estimated_pose, color="blue")
            self.plot_pose(pose=self.pose, color="red")
            plt.draw()
            plt.pause(0.05)

    @staticmethod
    def plot_pose(pose, color):
        """Add a drawing to the plot of a pose."""
        plt.plot([pose.x, pose.x], [pose.y, pose.y], "k-", alpha=0.8)
        plt.arrow(
            x=pose.x,
            y=pose.y,
            dx=0.1 * np.cos(pose.yaw),
            dy=0.1 * np.sin(pose.yaw),
            width=0.01,
            color=color,
        )

    def add_pose_to_plot(self, pose):
        """Add a drawing to the plot of the estimate and true poses."""
        plt.plot([self.pose.x, pose.x], [self.pose.y, pose.y], "k-", alpha=0.8)
        self.plot_pose(pose=pose, color="blue")
        self.plot_pose(pose=self.pose, color="red")
        plt.draw()
        plt.pause(0.05)


if __name__ == "__main__":

    class MeasureAngleFunc:
        """Find the y position given the state and control vectors."""

        def __init__(self, state_control: np.ndarray, feature: Feature):
            pos_x, pos_y, theta, _, _ = state_control
            delta_x = feature.x - pos_x
            delta_y = feature.y - pos_y
            self.angle = np.arctan2(delta_y, delta_x) - theta

        def __call__(self):
            """Make the method callable."""
            return self.angle

    logger.info(MeasureAngleFunc)
