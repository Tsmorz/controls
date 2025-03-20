"""Add a doc string to my files."""

import numpy as np
import pytest

from src.data_classes.lie_algebra import SE3


def test_se2() -> None:
    """Test that the SE2 class is correctly initialized."""
    # Arrange
    x = 1.0
    y = 2.0
    z = 3.0
    theta = 3.0

    # Act
    se3 = SE3(
        xyz=np.array([[x], [y], [z]]), roll_pitch_yaw=np.array([[0.0], [0.0], [theta]])
    )

    # Assert
    assert se3.x == x
    assert se3.y == y
    assert se3.yaw == theta


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
def test_se3_matmul(x, y, theta) -> None:
    """Test that the SE2 instances are correctly multiplied."""
    # Arrange

    se3_1 = SE3(
        xyz=np.array([[0.0], [0.0], [0.0]]),
        roll_pitch_yaw=np.array([[0.0], [0.0], [0.0]]),
    )
    se3_2 = SE3(
        xyz=np.array([[x], [y], [0]]), roll_pitch_yaw=np.array([[0.0], [0.0], [theta]])
    )

    # Act
    se3_3 = se3_1 @ se3_2

    # Assert
    np.testing.assert_array_almost_equal(se3_3.as_vector(), se3_2.as_vector())


def test_se2_matmul_fail() -> None:
    """Test that the SE2 instances are correctly multiplied."""
    # Arrange
    se3 = SE3(
        xyz=np.array([[0.0], [0.0], [0.0]]),
        roll_pitch_yaw=np.array([[0.0], [0.0], [0.0]]),
    )

    # Act and Assert
    with pytest.raises(ValueError):
        _ = se3 @ np.array([0])
