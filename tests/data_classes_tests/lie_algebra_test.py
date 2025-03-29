"""Add a doc string to my files."""

import numpy as np
import pytest

from ekf_slam_3d.data_classes.lie_algebra import SE3


@pytest.mark.parametrize(
    ("xyz", "rpy"),
    [
        (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])),
        (None, None),
    ],
)
def test_se2(xyz, rpy) -> None:
    """Test that the SE2 class is correctly initialized."""
    # Arrange
    if xyz is None:
        x, y, z = 0.0, 0.0, 0.0
    else:
        x, y, z = 1.0, 2.0, 3.0

    if rpy is None:
        roll, pitch, yaw = 0.0, 0.0, 0.0
    else:
        roll, pitch, yaw = 1.0, 2.0, 3.0

    # Act
    se3 = SE3(xyz=xyz, roll_pitch_yaw=rpy)

    # Assert
    np.testing.assert_almost_equal(se3.x, x)
    np.testing.assert_almost_equal(se3.y, y)
    np.testing.assert_almost_equal(se3.z, z)
    np.testing.assert_almost_equal(se3.roll, roll)
    np.testing.assert_almost_equal(se3.pitch, pitch)
    np.testing.assert_almost_equal(se3.yaw, yaw)


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
