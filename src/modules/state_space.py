"""Basic docstring for my module."""

from typing import Optional

import numpy as np

from src.modules.math_utils import matrix_exp


class StateSpace:
    """A discrete-time state-space model representation."""

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: Optional[np.ndarray] = None,
        D: Optional[np.ndarray] = None,
    ):
        """Initialize the state-space model.

        :param A: State transition matrix
        :param B: Control input matrix
        :param C: Observation matrix
        :param D: Direct transmission matrix
        """
        self.A = A
        self.B = B

        if C is None:
            C = np.eye(A.shape[0])
        self.C = C

        if D is None:
            D = np.zeros((C.shape[0], B.shape[1]))
        self.D = D


def continuous_to_discrete(state_space: StateSpace, dt: float) -> StateSpace:
    """Convert a continuous state space to discrete state space."""
    A = matrix_exp(state_space.A, dt)
    B = np.linalg.inv(state_space.A) @ (A - np.eye(A.shape[0])) @ state_space.B

    state_space.A = A
    state_space.B = B

    return state_space
