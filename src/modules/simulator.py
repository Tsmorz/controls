"""Basic docstring for my module."""

from typing import Optional

import numpy as np
from jax import numpy as jnp

from config.definitions import DEFAULT_DISCRETIZATION
from src.modules.state_space import StateSpaceLinear, StateSpaceNonlinear


class Simulator:
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

    def pos_x_func(state: np.ndarray, control: np.ndarray) -> jnp.ndarray:
        """Find the x position given the state and control vectors."""
        pos_x, pos_y, theta = state
        vel, theta_dot = control
        return vel * jnp.cos(theta) + pos_x

    def pos_y_func(state: np.ndarray, control: np.ndarray) -> jnp.ndarray:
        """Find the y position given the state and control vectors."""
        pos_x, pos_y, theta = state
        vel, theta_dot = control
        return vel * jnp.sin(theta) + pos_y

    def heading_func(state: np.ndarray, control: np.ndarray) -> jnp.ndarray:
        """Find the heading given the state and control vectors."""
        pos_x, pos_y, theta = state
        vel, theta_dot = control
        return jnp.array(theta + theta_dot)

    motion_model = [
        pos_x_func,
        pos_y_func,
        heading_func,
    ]

    return StateSpaceNonlinear(motion_model=motion_model)
