"""Basic docstring for my module."""

import copy
from typing import Callable, Optional

import numpy as np
from loguru import logger
from scipy.signal import cont2discrete

from config.definitions import EPSILON
from src.data_classes.state_space import StateSpaceData, plot_history


class StateSpaceLinear:
    """A discrete-time state-space model representation."""

    def __init__(
        self,
        A: np.ndarray,
        B: Optional[np.ndarray] = None,
        C: Optional[np.ndarray] = None,
        D: Optional[np.ndarray] = None,
    ):
        """Initialize the state-space model.

        :param A: State transition matrix
        :param B: Control input matrix
        :param C: Observation matrix
        :param D: Direct transmission matrix
        """
        if B is None:
            B = np.zeros((A.shape[0], 1))
        if A.shape[0] != B.shape[0]:
            msg = (
                f"A and B matrices must have the same number of rows. "
                f"{A.shape[0]} != {B.shape[0]}"
            )
            logger.error(msg)
            raise ValueError(msg)
        self.A = A
        self.B = B

        if C is None:
            C = np.eye(A.shape[0])
        self.C = C

        if D is None:
            D = np.zeros((C.shape[0], B.shape[1]))
        self.D = D

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Step the state-space model by one step.

        :param x: Current state
        :param u: Control input
        :return: Next state
        """
        return self.A @ x + self.B @ u

    def step_response(
        self, delta_t: float, plot_response: bool = True
    ) -> StateSpaceData:
        """Compute the step response of the state-space model.

        :param delta_t: Time step size
        :param plot_response: Whether to plot the step response
        :return: State history object
        """
        # Generate control input over 10 steps
        num_states = self.A.shape[0]
        num_inputs = self.B.shape[1]
        time = np.arange(0, 10, delta_t).tolist()

        control_input = len(time) * [np.ones((num_inputs, 1))]

        state_history = self.solve_trajectory(
            time=time,
            x0=np.zeros((num_states, 1)),
            control_input=control_input,
        )

        if plot_response:  # pragma: no cover
            plot_history(history=state_history, title="Step Response")

        return state_history

    def impulse_response(
        self, delta_t: float, plot_response: bool = True
    ) -> StateSpaceData:
        """Compute the step response of the state-space model.

        :param delta_t: Time step size
        :param plot_response: Whether to plot the step response
        :return: State history object
        """
        # Generate control input over 10 steps
        num_states = self.A.shape[0]
        num_inputs = self.B.shape[1]
        time = np.arange(0, 10, delta_t).tolist()

        control_input = len(time) * [np.zeros((num_inputs, 1))]
        control_input[0] = np.ones((num_inputs, 1))

        state_history = self.solve_trajectory(
            time=time,
            x0=np.zeros((num_states, 1)),
            control_input=control_input,
        )

        if plot_response:  # pragma: no cover
            plot_history(history=state_history, title="Impulse Response")

        return state_history

    def solve_trajectory(
        self, time: list[float], x0: np.ndarray, control_input: list[np.ndarray]
    ) -> StateSpaceData:
        """Solve the state-space model for given time, control input, and initial state.

        :param time: Time array
        :param x0: Initial state
        :param control_input: Control input array
        """
        state_history = StateSpaceData(control=control_input, time=time, state=[x0])
        x = x0
        for ii, _t in enumerate(time[:-1]):
            u = control_input[ii]
            x = self.step(x=x, u=u)
            state_history.state.append(x)
        return state_history

    def continuous_to_discrete(self, discretization_dt) -> None:
        """Convert a continuous state space to discrete state space."""
        system = (self.A, self.B, self.C, self.D)
        system_disc = cont2discrete(system, dt=discretization_dt, method="zoh")

        self.A = system_disc[0]
        self.B = system_disc[1]
        self.C = system_disc[2]
        self.D = system_disc[3]


class StateSpaceNonlinear:
    """A class for representing a nonlinear state-space model."""

    def __init__(
        self,
        motion_model: list[Callable],
        measurement_model: Optional[list[Callable]] = None,
    ):
        """Initialize a nonlinear state space model."""
        self.motion_model = motion_model
        self.measurement_model = measurement_model

    def linearize(self, x: np.ndarray, u: np.ndarray) -> StateSpaceLinear:
        """Linearize a list of callables.

        :param x: Current state
        :param u: Control input
        :return: Jacobian matrix
        """
        # linearize the motion model around the current state
        dfdx = np.zeros((len(self.motion_model), len(x)))
        for func_idx, func in enumerate(self.motion_model):
            for x_idx in range(len(x)):
                dfdx[func_idx, x_idx] = self._partial_derivative_x(
                    func=func, x=x, u=u, idx=x_idx
                )

        # linearize the motion model around the control input
        dfdu = np.zeros((len(self.motion_model), len(u)))
        for func_idx, func in enumerate(self.motion_model):
            for u_idx in range(len(u)):
                dfdu[func_idx, u_idx] = self._partial_derivative_u(
                    func=func, x=x, u=u, idx=u_idx
                )
        return StateSpaceLinear(A=dfdx, B=dfdu)

    @staticmethod
    def _partial_derivative_x(
        func: Callable, x: np.ndarray, u: np.ndarray, idx: int
    ) -> np.ndarray | float:
        state_copy1, state_copy2 = copy.deepcopy(x), copy.deepcopy(x)
        state_copy1[idx, 0] = state_copy1[idx, 0] - EPSILON
        state_copy2[idx, 0] = state_copy2[idx, 0] + EPSILON
        value = (func(state_copy2, u) - func(state_copy1, u)) / (2 * EPSILON)

        if isinstance(value, float):
            return value
        return value[0]

    @staticmethod
    def _partial_derivative_u(
        func: Callable, x: np.ndarray, u: np.ndarray, idx: int
    ) -> np.ndarray | float:
        control_copy1, control_copy2 = copy.deepcopy(u), copy.deepcopy(u)
        control_copy1[idx, 0] -= EPSILON
        control_copy2[idx, 0] += EPSILON
        value = (func(x, control_copy2) - func(x, control_copy1)) / (2 * EPSILON)
        if isinstance(value, float):
            return value
        return value[0]

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Step the state-space model by one step.

        :param x: Current state
        :param u: Control input
        :return: Next state
        """
        x_new = np.zeros_like(x)
        for ii, func in enumerate(self.motion_model):
            x_new[ii, 0] = func(x, u)[0]
        return x_new


# April 1 - April 16
# April 22 - April 30
# May 9 - May 18
