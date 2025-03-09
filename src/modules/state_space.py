"""Basic docstring for my module."""

from typing import Optional

import numpy as np

DEFAULT_DT = 1.0


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

        self.dt = DEFAULT_DT  # Default time step

    def continuous_to_discrete(self, delta_time: float = DEFAULT_DT) -> None:
        self.dt = delta_time

        A = matrix_exp(self.A, self.dt)
        B = np.linalg.inv(self.A) @ (A - np.eye(2)) @ self.B

        self.A = A
        self.B = B


def matrix_exp(matrix: np.ndarray, dt: float) -> np.ndarray:
    """Find the matrix exponential.

    :param matrix: Input matrix
    :param dt: Time step
    :return: Matrix exponential
    """
    d, U = np.linalg.eig(matrix)
    D = np.exp(d * dt) * np.eye(np.shape(matrix)[0])
    A = np.real(U @ D @ np.linalg.inv(U))
    return A
