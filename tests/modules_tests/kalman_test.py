"""Add a doc string to my files."""

import numpy as np

from config.definitions import DEFAULT_VARIANCE, MEASUREMENT_NOISE, PROCESS_NOISE
from ekf_slam.modules.kalman import KalmanFilter
from ekf_slam.modules.simulators import mass_spring_damper_model
from tests.conftest import TEST_DECIMALS_ACCURACY, TEST_DT


def test_kalman_filter_initialization() -> None:
    """Test the initialization of the Kalman filter."""
    ss = mass_spring_damper_model(discretization_dt=TEST_DT)

    exp_Q = PROCESS_NOISE * np.eye(2)
    exp_R = MEASUREMENT_NOISE * np.eye(2)
    initial_state = np.array([[1.0], [1.0]])
    initial_covariance = np.eye(2)

    kf = KalmanFilter(
        state_space=ss,
        initial_x=initial_state,
        initial_covariance=initial_covariance,
    )

    assert isinstance(kf, KalmanFilter)
    assert np.array_equal(kf.state_space.A, ss.A)
    assert np.array_equal(kf.state_space.B, ss.B)
    assert np.array_equal(kf.state_space.C, ss.C)
    assert np.array_equal(kf.Q, exp_Q)
    assert np.array_equal(kf.R, exp_R)
    assert np.array_equal(kf.cov, initial_covariance)
    assert np.array_equal(kf.x, initial_state)


def test_kalman_filter_predict_with_control_input() -> None:
    """Test that the next state and covariance is predicted correctly."""
    ss = mass_spring_damper_model(discretization_dt=TEST_DT)

    Q = DEFAULT_VARIANCE * np.eye(2)
    R = DEFAULT_VARIANCE * np.eye(1)
    initial_state = np.array([[1.0], [1.0]])
    initial_covariance = np.eye(2)

    kf = KalmanFilter(
        state_space=ss,
        process_noise=Q,
        measurement_noise=R,
        initial_x=initial_state,
        initial_covariance=initial_covariance,
    )

    control_input = np.array([[0.0]])
    expected_next_state = ss.step(x=kf.x, u=control_input)
    expected_next_covariance = ss.A @ kf.cov @ ss.A.T + Q

    kf.predict()

    assert np.allclose(kf.x, expected_next_state, atol=TEST_DECIMALS_ACCURACY)
    assert np.allclose(kf.cov, expected_next_covariance, atol=TEST_DECIMALS_ACCURACY)
