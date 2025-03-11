"""Basic docstring for my module."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.signal import cont2discrete

from config.definitions import DEFAULT_DT, FIG_SIZE, LEGEND_LOC
from src.data_classes.state_space_data import StateSpaceData


class StateSpace:
    """A discrete-time state-space model representation."""

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: Optional[np.ndarray] = None,
        D: Optional[np.ndarray] = None,
    ):
        """Initialize the state-space model.

        :param A: State transition matrix
        :param B: Control input matrix
        :param C: Observation matrix
        :param D: Direct transmission matrix
        """
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

        if plot_response:
            self.plot_history(history=state_history, title="Step Response")

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

        if plot_response:
            self.plot_history(history=state_history, title="Impulse Response")

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

    def plot_history(
        self, history: StateSpaceData, title: str = "State Space History"
    ) -> None:  # pragma: no cover
        """Plot the history of state space model.

        :param history: State history object
        :param title: Plot title
        :return: None
        """
        num_states = self.A.shape[0]
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=FIG_SIZE)
        plt.suptitle(title)

        for ii, ax in enumerate(axs):
            for state in range(num_states):
                data = [arr[state] for arr in history.state]
                if ii == 1:
                    data /= np.amax(np.abs(data))
                ax.step(history.time, data, label=f"$x_{state}$")
                ax.set_ylabel("State" if ii == 0 else "Normalized State")
            for u in range(history.control[0].shape[1]):
                control = [arr[u] for arr in history.control]
                if ii == 1:
                    control /= np.amax(np.abs(control))
                ax.step(history.time, control, label="control input")

        for ax in axs:
            ax.set_xlabel("Time (s)")
            ax.grid(True)
            ax.legend(loc=LEGEND_LOC)

        plt.show()
        plt.close()


def continuous_to_discrete(
    state_space: StateSpace, discretization_dt: float = DEFAULT_DT
) -> StateSpace:
    """Convert a continuous state space to discrete state space.

    :param state_space: Continuous state-space model
    :param discretization_dt: Desired discrete time step size
    :return: Discrete state-space model
    """
    system = (state_space.A, state_space.B, state_space.C, state_space.D)
    system_disc = cont2discrete(system, dt=discretization_dt, method="zoh")

    state_space.A = system_disc[0]
    state_space.B = system_disc[1]
    state_space.C = system_disc[2]
    state_space.D = system_disc[3]

    return state_space


def mass_spring_damper_model(
    mass: float = 0.5,
    spring_const: float = 10.0,
    damping: float = 0.8,
    discretization_dt: Optional[float] = None,
) -> StateSpace:  # pragma: no cover
    """Calculate a simple mass spring damper model.

    :param mass: Mass of the system
    :param spring_const: Spring constant
    :param damping: Damping coefficient
    :param discretization_dt: Desired discrete time step size
    :return: state-space model
    """
    model = StateSpace(
        A=np.array([[0.0, 1.0], [-spring_const / mass, -damping / mass]]),
        B=np.array([[0.0], [1.0 / mass]]),
    )

    if discretization_dt is not None:
        model = continuous_to_discrete(model, discretization_dt=discretization_dt)
    return model


if __name__ == "__main__":  # pragma: no cover
    """Run the main program with this function."""
    dt = 0.05
    ss = mass_spring_damper_model(discretization_dt=dt)
    ss.step_response(delta_t=dt, plot_response=True)
    ss.impulse_response(delta_t=dt, plot_response=True)
