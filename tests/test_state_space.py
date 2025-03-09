import numpy as np

from src.modules.state_space import StateSpace


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
