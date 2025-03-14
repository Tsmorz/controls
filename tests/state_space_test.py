import numpy as np

from config.definitions import DEFAULT_DT
from src.data_classes.state_space_data import StateSpaceData
from src.modules.simulator import mass_spring_damper_model
from src.modules.state_space import StateSpaceLinear


def test_state_space() -> None:
    """Test that the state space is initialized correctly."""
    # Arrange
    A = np.array([[1, 0], [0, 1]])
    B = np.array([[1], [0]])

    # Act
    ss = StateSpaceLinear(A, B)

    # Assert
    np.testing.assert_array_almost_equal(ss.A, A)
    np.testing.assert_array_almost_equal(ss.B, B)
    np.testing.assert_array_almost_equal(ss.C, np.array([[1, 0], [0, 1]]))
    np.testing.assert_array_almost_equal(ss.D, np.array([[0], [0]]))


def test_continuous_to_discrete() -> None:
    """Test that the continuous to discrete is working correctly."""
    # Arrange
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    dt = DEFAULT_DT

    # Act
    ss = StateSpaceLinear(A, B)
    ss.continuous_to_discrete(discretization_dt=dt)

    # Assert
    expected_A = np.array([[1.0, dt], [0.0, 1.0]])
    expected_B = np.array([[dt**2 / 2], [dt]])
    np.testing.assert_array_almost_equal(ss.A, expected_A)
    np.testing.assert_array_almost_equal(ss.B, expected_B)


def test_state_space_data_append():
    """Test that the state space data can append results."""
    # Arrange
    data = StateSpaceData()

    # Act
    data.append_step(t=0.1, x=np.array([0.1, 0.2]), cov=np.eye(2), u=np.array([0.3]))

    # Assert
    np.testing.assert_array_almost_equal(data.time, [0.1])
    np.testing.assert_array_almost_equal(data.state, [np.array([0.1, 0.2])])
    np.testing.assert_array_almost_equal(data.covariance, [np.array([[1, 0], [0, 1]])])
    np.testing.assert_array_almost_equal(data.control, [np.array([0.3])])


def test_state_space_step() -> None:
    """Test that the state space step method works."""
    # Arrange
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    ss = StateSpaceLinear(A, B)

    x = np.array([0.1, 0.2])
    u = np.array([0.3])
    exp_x = ss.A @ x + ss.B @ u

    # Act
    x = ss.step(x, u)

    # Assert
    np.testing.assert_array_almost_equal(x, exp_x)


def test_state_space_incorrect_dims() -> None:
    """Test that the state space raises an error for incompatible dimensions."""
    # Arrange
    A = np.eye(2)
    B = np.array([[1, 2]])

    # Act and Assert
    with np.testing.assert_raises(ValueError):
        StateSpaceLinear(A, B)


def test_step_response() -> None:
    """Test that the step response method works correctly."""
    # Arrange
    dt = 0.05
    ss = mass_spring_damper_model(discretization_dt=dt)

    # Act
    data = ss.step_response(dt, plot_response=False)

    # Assert
    assert isinstance(data, StateSpaceData)
    assert len(data.time) > 0
    np.testing.assert_array_almost_equal(data.time[-1], 10 - dt)
    assert len(data.state) > 0
    np.testing.assert_array_almost_equal(data.state[0], np.array([[0], [0]]))
    assert len(data.covariance) == 0
    assert data.control[0] == np.ones((1, 1))
    assert data.control[-1] == np.ones((1, 1))


def test_impulse_response() -> None:
    """Test that the step response method works correctly."""
    # Arrange
    dt = 0.05
    ss = mass_spring_damper_model(discretization_dt=dt)

    # Act
    data = ss.impulse_response(dt, plot_response=False)

    # Assert
    assert isinstance(data, StateSpaceData)
    assert len(data.time) > 0
    np.testing.assert_almost_equal(data.time[-1], 10 - dt)
    assert len(data.state) > 0
    np.testing.assert_array_almost_equal(data.state[0], np.array([[0], [0]]))
    assert len(data.covariance) == 0
    assert data.control[0] == np.ones((1, 1))
    assert data.control[-1] == np.zeros((1, 1))
