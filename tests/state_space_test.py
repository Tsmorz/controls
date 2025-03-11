import numpy as np

from config.definitions import DEFAULT_DT
from src.data_classes.state_space_data import StateSpaceData
from src.modules.state_space import StateSpace, continuous_to_discrete


def test_state_space() -> None:
    """Test that the state space is initialized correctly."""
    # Arrange
    A = np.eye(2)
    B = np.array([[1], [0]])

    # Act
    ss = StateSpace(A, B)

    # Assert
    np.testing.assert_allclose(ss.A, A)
    np.testing.assert_allclose(ss.B, B)
    np.testing.assert_allclose(ss.C, np.eye(2))
    np.testing.assert_allclose(ss.D, np.zeros((2, 1)))


def test_continuous_to_discrete() -> None:
    """Test that the continuous to discrete is working correctly."""
    # Arrange
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    ss = StateSpace(A, B)
    dt = DEFAULT_DT

    # Act
    discrete_ss = continuous_to_discrete(ss, dt)

    # Assert
    expected_A = np.array([[1.0, dt], [0.0, 1.0]])
    expected_B = np.array([[dt**2 / 2], [dt]])
    np.testing.assert_array_almost_equal(discrete_ss.A, expected_A)
    np.testing.assert_array_almost_equal(discrete_ss.B, expected_B)


def test_state_space_data_append():
    """Test that the state space data can append results."""
    # Arrange
    data = StateSpaceData()

    # Act
    data.append_step(t=0.1, x=np.array([0.1, 0.2]), cov=np.eye(2), u=np.array([0.3]))

    # Assert
    np.testing.assert_array_almost_equal(data.time, [0.1])
    np.testing.assert_array_almost_equal(data.state, [np.array([0.1, 0.2])])
    np.testing.assert_array_almost_equal(data.covariance, [np.eye(2)])
    np.testing.assert_array_almost_equal(data.control, [np.array([0.3])])
