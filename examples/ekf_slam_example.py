"""Basic docstring for my module."""

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from src.data_classes.lie_algebra import SE3
from src.data_classes.map import Feature, make_random_map_planar
from src.data_classes.sensors import Bearing, Distance, DistanceAndBearing, SensorType
from src.data_classes.slam import PoseMap
from src.modules.extended_kalman import ExtendedKalmanFilter
from src.modules.state_space import StateSpaceNonlinear


def state_to_se3(state: np.ndarray) -> SE3:
    """Map the state vector to SE2."""
    return SE3(xyz=state[0:3], roll_pitch_yaw=state[3:6])


def motion_eqns(state_control: np.ndarray) -> np.ndarray:
    """Define the equations of motion.

    :param state_control: the state and control vectors
    :return: the state vector after applying the motion equations
    """
    vel, omega = state_control[-2:]
    pose = state_to_se3(state_control[:6, 0])

    state_vec = np.zeros_like(state_control)
    state_vec[0, 0] = vel * np.cos(pose.yaw) * np.cos(pose.pitch) + pose.x
    state_vec[1, 0] = vel * np.sin(pose.yaw) * np.cos(pose.pitch) + pose.y
    state_vec[2, 0] = vel * np.sin(pose.pitch) + pose.z
    state_vec[3, 0] = pose.roll
    state_vec[4, 0] = pose.pitch
    state_vec[5, 0] = pose.yaw + omega
    return state_control


def measurement_eqns(state_control: np.ndarray, feature: Feature) -> np.ndarray:
    """Define the equations of measurement.

    :param state_control: the state and control vectors
    :param feature: the feature to be measured
    :return: the measurement vector at the current state
    """
    pose = state_to_se3(state_control[:6, 0])
    delta_x = feature.x - pose.x
    delta_y = feature.y - pose.y
    delta_z = feature.z - pose.z

    distance = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
    angle = np.arctan2(delta_y, delta_x) - pose.yaw

    measurement = np.zeros((2, 1))
    measurement[0, 0] = distance
    measurement[1, 0] = angle
    return measurement


def pos_x_func(state_control: np.ndarray) -> float | np.ndarray:
    """Find the x position given the state and control vectors."""
    vel, omega = state_control[-2:]
    pose = state_to_se3(state_control[:6, 0])
    return vel * np.cos(pose.yaw) * np.cos(pose.pitch) + pose.x


def pos_y_func(state_control: np.ndarray) -> float | np.ndarray:
    """Find the y position given the state and control vectors."""
    vel, omega = state_control[-2:]
    pose = state_to_se3(state_control[:6, 0])
    return vel * np.sin(pose.yaw) * np.cos(pose.pitch) + pose.y


def pos_z_func(state_control: np.ndarray) -> float | np.ndarray:
    """Find the y position given the state and control vectors."""
    vel, omega = state_control[-2:]
    pose = state_to_se3(state_control[:6, 0])
    return vel * np.sin(pose.pitch) + pose.z


def roll_func(state_control: np.ndarray) -> float | np.ndarray:
    """Find the heading given the state and control vectors."""
    pose = state_to_se3(state_control[:6, 0])
    return pose.roll


def pitch_func(state_control: np.ndarray) -> float | np.ndarray:
    """Find the heading given the state and control vectors."""
    pose = state_to_se3(state_control[:6, 0])
    return pose.pitch


def yaw_func(state_control: np.ndarray) -> float | np.ndarray:
    """Find the heading given the state and control vectors."""
    vel, omega = state_control[-2:]
    pose = state_to_se3(state_control[:6, 0])
    return np.array(pose.yaw + omega)


def measure_range_func(
    state_control: np.ndarray, feature: Feature
) -> float | np.ndarray:
    """Find the x position given the state and control vectors."""
    pose = state_to_se3(state_control[:6, 0])
    delta_x = feature.x - pose.x
    delta_y = feature.y - pose.y
    delta_z = feature.z - pose.z
    distance = np.sqrt(delta_x**2 + delta_y**2 + delta_z)
    return distance


def measure_angle_func(
    state_control: np.ndarray, feature: Feature
) -> float | np.ndarray:
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


def pipeline() -> None:
    """Run the EKF for SLAM."""
    logger.info("Running Extended Kalman Filter pipeline.")

    robot = robot_model()
    robot_pose = SE3(xyz=np.zeros((3,)), roll_pitch_yaw=np.zeros((3,)))
    pose_map = PoseMap(pose=robot_pose)
    ekf = ExtendedKalmanFilter(
        state_space_nonlinear=robot,
        initial_x=pose_map.pose.as_vector(),
        initial_covariance=3 * np.eye(robot_pose.as_vector().shape[0]),
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
        control_input = np.array([[linear_vel], [angular_vel]])
        sim.step(u=control_input)
        ekf.predict(u=control_input)
        pose_map.from_vector(ekf_state=ekf.x)

        if (time / 20) % 1 == 0 and time != 0:
            for ii, feat in enumerate(true_map.features):
                feature_new = Feature(x=time, y=time, id=ii * time)
                meas = sim.get_measurement(
                    feature=feat,
                    sensor_type=SensorType.DISTANCE_AND_BEARING,
                )
                logger.info(f"Measurement={meas}")
                pose_map.map.append_feature(feature=feature_new)

                ekf.update(z=meas.as_vector(), u=control_input, measurement_args=feat)
                if meas.type == SensorType.DISTANCE_AND_BEARING:
                    add_measurement_to_plot(meas, state=ekf.x)

        logger.info(f"state: {ekf.x.T}")
        robot_pose = state_to_se3(state=ekf.x)
        sim.append_estimate(estimated_pose=robot_pose, plot_pose=True)

    plt.show()
    plt.close()


if __name__ == "__main__":
    pipeline()
