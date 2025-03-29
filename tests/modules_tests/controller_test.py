"""Add a doc string to my files."""

import numpy as np

from ekf_slam.modules.controller import full_state_feedback, get_control_input
from ekf_slam.modules.simulators import mass_spring_damper_model


def test_get_control_input() -> None:
    """Test that the control input is calculated correctly."""
    # Arrange
    x = np.array([[2.0], [2.0]])
    desired = np.array([[1.0], [1.0]])
    gain_matrix = np.array([[1.0, 1.0]])
    limit = 10.0

    # Act
    control = get_control_input(x, desired, gain_matrix, limit)

    # Assert
    np.testing.assert_array_almost_equal(control, np.array([-sum(x - desired)]))


def test_full_state_feedback() -> None:
    """Test that the full state feedback is calculated correctly."""
    # Arrange
    state_space = mass_spring_damper_model()
    desired_eigenvalues = np.array([1.0, 2.0])

    # Act
    gain_matrix = full_state_feedback(state_space, desired_eigenvalues)
    A_prime = state_space.A - state_space.B @ gain_matrix

    # Assert
    new_eigenvalues = np.linalg.eigvals(A_prime)
    np.testing.assert_array_almost_equal(new_eigenvalues, desired_eigenvalues)
