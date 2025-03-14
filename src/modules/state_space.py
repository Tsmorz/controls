"""Basic docstring for my module."""

import copy
from typing import Callable, Optional

import numpy as np
from loguru import logger
from scipy.signal import cont2discrete

from config.definitions import EPSILON
from src.data_classes.state_space_data import StateSpaceData, plot_history


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


def mass_spring_damper_discrete(
    mass: float = 0.5,
    spring_const: float = 20.0,
    damping: float = 0.4,
    discretization_dt: float = 0.0,
) -> StateSpaceLinear:  # pragma: no cover
    """Calculate a simple mass spring damper model.

    :param mass: Mass of the system
    :param spring_const: Spring constant
    :param damping: Damping coefficient
    :param discretization_dt: Desired discrete time step size
    :return: state-space model
    """
    model = StateSpaceLinear(
        A=np.array([[0.0, 1.0], [-spring_const / mass, -damping / mass]]),
        B=np.array([[0.0], [1.0 / mass]]),
    )
    model.continuous_to_discrete(discretization_dt)
    return model


class StateSpaceNonlinear:
    """A class for representing a nonlinear state-space model."""

    def __init__(
        self,
        f: list[Callable],
        h: Optional[list[Callable]] = None,
    ):
        """Initialize a nonlinear state space model."""
        self.f = f
        self.h = h

    def linearize(self, x: np.ndarray, u: np.ndarray) -> StateSpaceLinear:
        """Linearize a list of callables.

        :param x: Current state
        :param u: Control input
        :return: Jacobian matrix
        """
        jacobian = np.zeros((len(self.f), len(x) + len(u)))
        for ii, func in enumerate(self.f):
            for jj, _s in enumerate(x):
                jacobian[ii, jj] = self._partial_derivative(func=func, x=x, u=u, jj=jj)
        A = jacobian[:, : len(x)]
        B = jacobian[:, len(x) :]
        return StateSpaceLinear(A, B)

    @staticmethod
    def _partial_derivative(
        func: Callable, x: np.ndarray, u: np.ndarray, jj: int
    ) -> np.ndarray:
        state_copy1, state_copy2 = copy.deepcopy(x), copy.deepcopy(x)
        state_copy1[jj, 0] -= EPSILON
        state_copy2[jj, 0] += EPSILON
        value = (func(state_copy2, u) - func(state_copy1, u)) / (2 * EPSILON)
        return value[0]

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Step the state-space model by one step.

        :param x: Current state
        :param u: Control input
        :return: Next state
        """
        ss = self.linearize(x=x, u=u)
        return ss.A @ x + ss.B @ u


if __name__ == "__main__":  # pragma: no cover
    """Run the main program with this function."""
    dt = 0.05
    ss_model = mass_spring_damper_discrete(discretization_dt=dt)
    ss_model.step_response(delta_t=dt, plot_response=True)
    ss_model.impulse_response(delta_t=dt, plot_response=True)

# April 1 - April 16
# April 22 - April 30
# May 9 - May 18
