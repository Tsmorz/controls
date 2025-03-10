"""Add a doc string to my files."""

import numpy as np


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
