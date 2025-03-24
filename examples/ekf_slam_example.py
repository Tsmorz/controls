"""Basic docstring for my module."""

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.pyplot import Axes, Figure

from config.definitions import VECTOR_LENGTH
from src.data_classes.lie_algebra import SE3, state_to_se3
from src.data_classes.map import Feature, make_random_map_planar
from src.data_classes.sensors import (
    DistanceAndBearing,
    Dynamics,
)
from src.data_classes.slam import Map, PoseMap
from src.modules.kalman_extended import ExtendedKalmanFilter
from src.modules.state_space import StateSpaceNonlinear


def motion_eqns(state_control: np.ndarray) -> np.ndarray:
    """Define the equations of motion.

    :param state_control: the state and control vectors
    :return: the state vector after applying the motion equations
    """
    vel, omega = state_control[-2:]
    pose = state_to_se3(state_control[:6, 0])

    state_vec = np.zeros((6, 1))
    state_vec[0, 0] = vel[0] * np.cos(pose.yaw) * np.cos(pose.pitch) + pose.x
    state_vec[1, 0] = vel[0] * np.sin(pose.yaw) * np.cos(pose.pitch) + pose.y
    state_vec[2, 0] = vel[0] * np.sin(pose.pitch) + pose.z
    state_vec[3, 0] = pose.roll
    state_vec[4, 0] = pose.pitch
    state_vec[5, 0] = pose.yaw + omega[0]
    return state_vec


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


def robot_model() -> StateSpaceNonlinear:
    """Create a StateSpaceNonlinear model of a wheeled robot."""
    return StateSpaceNonlinear(motion_model=Dynamics)


def add_measurement_to_plot(
    fig: tuple[Figure, Axes],
    distances: np.ndarray,
    bearings: np.ndarray,
    state: np.ndarray,
) -> None:
    """Plot the simulation results."""
    rays = []
    pose = state_to_se3(state=state)
    for r, b in zip(distances, bearings):
        x1, x2 = pose.x, pose.x + r * np.cos(pose.yaw + b)
        y1, y2 = pose.y, pose.y + r * np.sin(pose.yaw + b)
        (m,) = plt.plot([x1, x2], [y1, y2], "k-", alpha=0.2)
        rays.append(m)
    plt.axis("equal")
    plt.pause(0.1)
    for ray in rays:
        ray.remove()
    fig[0].canvas.draw()
    fig[0].canvas.flush_events()


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

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        plt.axis("equal")
        plt.grid(True)
        plt.xlabel("x position")
        plt.ylabel("y position")
        plt.title("Robot Localization")
        for feature in self.map.features:
            ax.plot(feature.x, feature.y, "k*")
        self.sim_plot = (fig, ax)

    def step(self, u: np.ndarray) -> SE3:
        """Predict the next state and error covariance.

        :param u: Control input
        """
        scale = np.diag(self.Q)
        scale = np.reshape(scale, (self.Q.shape[0], 1))
        noise = np.random.normal(loc=0.0, scale=scale, size=(self.Q.shape[0], 1))
        x = self.state_space_nl.step(x=self.pose.as_vector(), u=u) + noise

        # set z pos, roll, and pitch to zero
        x[2, 0], x[3, 0], x[4, 0] = 0.0, 0.0, 0.0

        self.pose = SE3(xyz=x[0:3], roll_pitch_yaw=x[3:6])
        return self.pose

    def append_estimate(self, estimated_pose: SE3) -> None:
        """Update the state estimate based on an estimated pose."""
        self.history.append((estimated_pose, self.pose))
        self.plot_pose(pose=estimated_pose, color="blue")
        self.plot_pose(pose=self.pose, color="red")
        plt.axis("equal")
        plt.draw()
        plt.pause(0.1)

    def plot_pose(self, pose, color):
        """Add a drawing to the plot of a pose."""
        fig, ax = self.sim_plot
        ax.plot([pose.x, pose.x], [pose.y, pose.y], "k-", alpha=0.8)
        ax.arrow(
            x=pose.x,
            y=pose.y,
            dx=VECTOR_LENGTH * np.cos(pose.yaw),
            dy=VECTOR_LENGTH * np.sin(pose.yaw),
            width=0.01,
            color=color,
        )
        fig.show()


def pipeline() -> None:
    """Run the EKF for SLAM."""
    logger.info("Running Extended Kalman Filter pipeline.")

    robot = StateSpaceNonlinear(motion_model=Dynamics)
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
        initial_pose=pose_map.pose,
        sim_map=make_random_map_planar(num_features=8, dim=(25, 25)),
        steps=200,
    )

    linear_vel = 0.5
    for time, angular_vel in enumerate(sim.controls):
        control_input = np.array([[linear_vel], [angular_vel]])
        sim.step(u=control_input)
        ekf.predict(u=control_input)
        pose_map.from_vector(ekf_state=ekf.x)

        if (time / 15) % 1 == 0 and time != 0:
            meas = DistanceAndBearing(
                state=sim.pose.as_vector(), features=sim.map.features
            )

            logger.info(f"Measurement={meas}")
            ekf.update(z=meas, u=control_input, measurement_args=sim.map.features)

            add_measurement_to_plot(
                fig=sim.sim_plot,
                distances=meas.distance,
                bearings=meas.bearing,
                state=ekf.x,
            )
            for feat in sim.map.features:
                pose_map.map.append_feature(feature=feat)

        x_str = np.array2string(ekf.x.T, precision=2, floatmode="fixed")
        logger.info(f"state: {x_str}")
        robot_pose = state_to_se3(state=ekf.x)
        sim.append_estimate(estimated_pose=robot_pose)

    plt.show()
    plt.close()


if __name__ == "__main__":
    pipeline()
