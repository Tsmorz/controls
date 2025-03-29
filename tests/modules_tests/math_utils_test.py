"""Add a doc string to my files."""

import numpy as np
import pytest

from ekf_slam.modules.math_utils import matrix_exponential


@pytest.mark.parametrize("t", [1.0, 0.1, 0.01])
def test_matrix_exponential_jordan(t):
    """Test matrix exponential function with different time lengths."""
    # Arrange
    matrix = np.array(
        [
            [0, 1],
            [0, 0],
        ]
    )
    expected = np.array(
        [
            [1, t],
            [0, 1],
        ]
    )

    # Act
    mat_exp = matrix_exponential(matrix, t=t)

    # Assert
    np.testing.assert_array_almost_equal(mat_exp, expected, decimal=3)


@pytest.mark.parametrize("t", [1.0, 0.1, 0.01])
def test_matrix_exponential_diagonal(t: float) -> None:
    """Test matrix exponential function with different time lengths."""
    # Arrange
    matrix = np.eye(5)
    expected = np.exp(t) * matrix

    # Act
    mat_exp = matrix_exponential(matrix, t=t)

    # Assert
    np.testing.assert_array_almost_equal(mat_exp, expected, decimal=3)


def test_matrix_exponential_nonsquare() -> None:
    """Test matrix exponential function with different time lengths."""
    # Arrange
    matrix_nonsquare = np.ones((3, 2))

    # Act / Assert
    with np.testing.assert_raises(ValueError):
        _ = matrix_exponential(matrix_nonsquare)
