"""Basic docstring for my module."""

from typing import Optional

import numpy as np

from src.modules.state_space import (
    StateSpaceLinear,
)


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
