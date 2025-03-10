"""Basic docstring for my module."""

from typing import Optional

import numpy as np
from loguru import logger

from config.definitions import DEFAULT_NUM_STEPS


class LQRController:
    """Discrete-time LQR Controller with finite horizon."""

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        num_steps: int = DEFAULT_NUM_STEPS,
    ):
        """Initialize the LQR controller.

        :param A: State matrix
        :param B: Input matrix
        :param Q: State cost matrix
        :param R: Control cost matrix
        :param num_steps: Time horizon
        """
        if A.shape[0] != B.shape[0]:
            msg = "A and B must have the same number of rows"
            logger.error(msg)
            raise ValueError(msg)
        self.A = A
        self.B = B

        if Q is None:
            Q = np.eye(A.shape[0])  # Default Q matrix
        if R is None:
            R = np.eye(B.shape[1])  # Default R matrix

        self.Q = Q
        self.R = R

        self.num_steps = num_steps
        self.K = self.compute_finite_horizon_lqr()

    def compute_finite_horizon_lqr(self):
        """Compute the finite-horizon LQR gains using backward recursion.

        :return: List of state feedback gains for each time step.
        """
        cost = self.Q  # Initialize terminal cost
        gain_list = []

        for _ in range(self.num_steps):
            gain = np.linalg.inv(self.B.T @ cost @ self.B + self.R) @ (
                self.B.T @ cost @ self.A
            )
            gain_list.insert(0, gain)  # Store gain for each time step
            cost = self.Q + self.A.T @ cost @ (self.A - self.B @ gain)

        return gain_list

    def get_control_gain(self, t):
        """Get the LQR gain at time step t.

        :param t: Current time step
        :return: State feedback gain K
        """
        if t >= self.num_steps:
            return self.K[-1]  # Use the last computed gain after horizon
        return self.K[t]
