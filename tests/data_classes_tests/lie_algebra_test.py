"""Add a doc string to my files."""

import numpy as np
import pytest

from src.data_classes.lie_algebra import SE2


def test_se2() -> None:
    """Test that the SE2 class is correctly initialized."""
    # Arrange
    x = 1.0
    y = 2.0
    theta = 3.0

    # Act
    se2 = SE2(x, y, theta)

    # Assert
    assert se2.x == x
    assert se2.y == y
    assert se2.theta == theta


@pytest.mark.parametrize(
    ("x", "y", "theta"),
    [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, np.pi),
        (0.0, 0.0, -np.pi),
        (0.0, 0.0, np.pi / 4),
        (0.0, 0.0, -np.pi / 4),
    ],
)
def test_se2_matmul(x, y, theta) -> None:
    """Test that the SE2 instances are correctly multiplied."""
    # Arrange
    se2_1 = SE2(x=0, y=0, theta=0)
    se2_2 = SE2(x=x, y=y, theta=theta)

    # Act
    se2_3 = se2_1 @ se2_2

    # Assert
    np.testing.assert_array_almost_equal(se2_3.as_vector(), se2_2.as_vector())


def test_se2_matmul_fail() -> None:
    """Test that the SE2 instances are correctly multiplied."""
    # Arrange
    se2 = SE2(x=0, y=0, theta=0)

    # Act and Assert
    with pytest.raises(ValueError):
        _ = se2 @ np.array([0])
