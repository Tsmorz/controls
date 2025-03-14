"""Basic docstring for my module."""

from typing import Callable, Optional

import numpy as np
from jax import grad
from jax import numpy as jnp
from loguru import logger

from config.definitions import LOG_DECIMALS, MEASUREMENT_NOISE, PROCESS_NOISE
from src.modules.math_utils import symmetrize_matrix
from src.modules.state_space import (
    StateSpace,
)

jnp.set_printoptions(precision=LOG_DECIMALS)


class StateSpaceNonlinear:
    """A class for representing a nonlinear state-space model."""

    def __init__(
        self,
        f: list[Callable],
        h: Optional[list[Callable]] = None,
    ):
        """Initialize a nonlinear state space model."""
        self.f = f
        self.h = h

    def linearize(self, x: np.ndarray) -> StateSpace:
        """Linearize a list of callables.

        :return: Jacobian matrix
        """
        jacobian = np.zeros((len(self.f), len(x)))
        for ii, f in enumerate(self.f):
            grad_f = grad(f, argnums=(0, 1, 2))
            jacobian[ii, :] = grad_f(x[0, 0], x[1, 0], x[2, 0])
        return StateSpace(A=jacobian, B=jacobian)

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Step the state-space model by one step.

        :param x: Current state
        :param u: Control input
        :return: Next state
        """
        ss = self.linearize(x=x)
        return ss.A @ x + ss.B @ u


class ExtendedKalmanFilter:
    """Kalman filter implementation."""

    def __init__(
        self,
        state_space_nonlinear: StateSpaceNonlinear,
        process_noise: np.ndarray,
        measurement_noise: np.ndarray,
        initial_x: np.ndarray,
        initial_covariance: np.ndarray,
    ) -> None:
        """Initialize the Kalman Filter.

        :param state_space_nonlinear: nonlinear state space model
        :param process_noise: Process noise covariance
        :param measurement_noise: Measurement noise covariance
        :param initial_covariance: Initial error covariance
        :return: None
        """
        self.ss_nl = state_space_nonlinear
        self.Q: np.ndarray = process_noise
        self.R: np.ndarray = measurement_noise
        self.x: np.ndarray = initial_x
        self.cov: np.ndarray = initial_covariance

    def predict(self, u: Optional[np.ndarray] = None) -> None:
        """Predict the next state and error covariance.

        :param u: Control input
        """
        state_space = self.ss_nl.linearize(self.x)

        if u is None:
            u = np.zeros((state_space.B.shape[1], 1))
        self.x = state_space.step(x=self.x, u=u)
        self.cov = state_space.A @ self.cov @ state_space.A.T + self.Q
        self.cov = symmetrize_matrix(self.cov)

    def update(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Update the state estimate with measurement z.

        :param z: Measurement
        :return: Updated state estimate and state covariance
        """
        state_space = self.ss_nl.linearize(self.x)

        y = z - state_space.C @ self.x  # Measurement residual
        S = state_space.C @ self.cov @ state_space.C.T + self.R  # Innovation covariance
        K = self.cov @ state_space.C.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        self.cov = (np.eye(self.cov.shape[0]) - K @ state_space.C) @ self.cov
        self.cov = symmetrize_matrix(self.cov)
        return self.x.copy(), self.cov.copy()


if __name__ == "__main__":

    def heading_func(pos_x: float, pos_y: float, theta: float) -> jnp.ndarray:
        """Find the heading given x1 and x2."""
        return jnp.array(theta)

    def pos_x_func(pos_x: float, pos_y: float, theta: float) -> jnp.ndarray:
        """Find the x velocity given x1 and x2."""
        return jnp.cos(theta) + pos_x

    def pos_y_func(pos_x: float, pos_y: float, theta: float) -> jnp.ndarray:
        """Find the y velocity given x1 and x2."""
        return jnp.sin(theta) + pos_y

    ss_nl = StateSpaceNonlinear(f=[heading_func, pos_x_func, pos_y_func])
    ekf = ExtendedKalmanFilter(
        state_space_nonlinear=ss_nl,
        process_noise=PROCESS_NOISE * np.eye(3),
        measurement_noise=MEASUREMENT_NOISE * np.eye(3),
        initial_x=np.array([[0.0], [0.0], [0.0]]),
        initial_covariance=5 * np.eye(3),
    )
    ekf.predict()
    ekf.update(z=np.array([[0.0], [0.0], [0.0]]))

    logger.info(f"Jacobian:\n{5.0}")
