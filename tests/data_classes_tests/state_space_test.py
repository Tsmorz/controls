"""Add a doc string to my files."""

import numpy as np
import pytest

from config.definitions import DEFAULT_DT
from src.data_classes.pose import SE2
from src.data_classes.state_space import StateSpaceData
from src.modules.simulator import mass_spring_damper_model, robot_model
from src.modules.state_space import StateSpaceLinear, StateSpaceNonlinear
from tests.conftest import TEST_DECIMALS_ACCURACY


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


def test_state_space_nonlinear() -> None:
    """Test the initialization of the Kalman filter."""

    def test_func(xu: np.ndarray) -> np.ndarray:
        return xu[0, 0] ** 2 + xu[1, 0]

    motion_model = [test_func]
    measurement_model = [test_func]
    state_space_nl = StateSpaceNonlinear(
        motion_model=motion_model, measurement_model=measurement_model
    )

    state = np.array([[3.0], [2.0]])
    A, B = state_space_nl.linearize(model=motion_model, x=state, u=np.zeros((2, 1)))
    state_space = StateSpaceLinear(A, B)

    exp_A = np.array([[2 * state[0, 0], 1]])
    np.testing.assert_array_almost_equal(
        state_space.A, exp_A, decimal=TEST_DECIMALS_ACCURACY
    )


@pytest.mark.parametrize(
    ("vel", "theta"),
    [
        (-0.5, -0.5),
        (-1.0, -1.0),
        (-1.5, -1.5),
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (1.5, 1.5),
    ],
)
def test_state_space_nonlinear_robot_model(vel: float, theta: float) -> None:
    """Test the initialization of the Kalman filter."""
    # Arrange
    robot = robot_model()
    pose = SE2(x=0.0, y=0.0, theta=theta)
    u = np.array([[vel], [0.0]])

    # Act
    A, B = robot.linearize(model=robot.motion_model, x=pose.as_vector(), u=u)
    state_space = StateSpaceLinear(A, B)

    # Assert
    exp_A = np.array(
        [
            [1.0, 0.0, -u[0, 0] * np.sin(pose.theta + u[1, 0])],
            [0.0, 1.0, u[0, 0] * np.cos(pose.theta + u[1, 0])],
            [0.0, 0.0, 1.0],
        ]
    )
    exp_B = np.array(
        [
            [np.cos(pose.theta), 0.0],
            [np.sin(pose.theta), 0.0],
            [0.0, 1.0],
        ]
    )

    np.testing.assert_array_almost_equal(
        state_space.A, exp_A, decimal=TEST_DECIMALS_ACCURACY
    )
    np.testing.assert_array_almost_equal(
        state_space.B, exp_B, decimal=TEST_DECIMALS_ACCURACY
    )
