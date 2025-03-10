"""Basic docstring for my module."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
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
        self.A = A
        self.B = B

        if C is None:
            C = np.eye(A.shape[0])
        self.C = C

        if D is None:
            D = np.zeros((C.shape[0], B.shape[1]))
        self.D = D

    def step(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        """Step the state-space model by one step.

        :param x: Current state
        :param u: Control input
        :return: Next state
        """
        if u is None:
            u = np.zeros((self.B.shape[1], 1))

        return self.A @ x + self.B @ u

    def step_response(self, dt: float, plot_response: bool = True) -> StateSpaceData:
        """Compute the step response of the state-space model.

        :param dt: Time step size
        :param plot_response: Whether to plot the step response
        :return: State history object
        """
        x = np.zeros((self.A.shape[0], 1))

        # Generate control input over 10 steps
        time = np.arange(0, 10, dt)
        num_steps = len(time)
        control_input = num_steps * [np.zeros((self.B.shape[1], 1))]
        control_input[0] = np.ones((self.B.shape[1], 1))

        state_history = StateSpaceData(control=control_input, time=time, state=[x])
        for ii, _t in enumerate(time[:-1]):
            u = control_input[ii]
            x = self.step(x=x, u=u)
            state_history.state.append(x)

        if plot_response:
            self.plot_history(history=state_history)

        return state_history

    def plot_history(self, history: StateSpaceData) -> None:
        """Plot the history of state space model.

        :param history: State history object
        :return: None
        """
        num_states = self.A.shape[0]
        fig, axs = plt.subplots(num_states, 1, sharex=True, figsize=FIG_SIZE)
        plt.suptitle("Step Response")

        ax = axs[0]
        for state in range(num_states):
            ax.step(
                history.time,
                [arr[state] for arr in history.state],
                label=f"$x_{state}$",
            )
            ax.set_ylabel("State")

        ax = axs[1]
        for state in range(num_states):
            data = [arr[state] for arr in history.state]
            data /= np.max(np.abs(data))
            ax.step(history.time, data, label=f"$x_{state}$")
            ax.set_ylabel("Normalized State")

        for state in range(num_states):
            ax = axs[state]
            ax.set_xlabel("Time (s)")
            ax.grid(True)
            ax.legend(loc=LEGEND_LOC)

        plt.show()
        plt.close()


def continuous_to_discrete(
    state_space: StateSpace, dt: float = DEFAULT_DT
) -> StateSpace:
    """Convert a continuous state space to discrete state space.

    :param state_space: Continuous state-space model
    :param dt: Desired discrete time step size
    :return: Discrete state-space model
    """
    system = (state_space.A, state_space.B, state_space.C, state_space.D)
    system_disc = cont2discrete(system, dt=dt, method="zoh")

    state_space.A = system_disc[0]
    state_space.B = system_disc[1]
    state_space.C = system_disc[2]
    state_space.D = system_disc[3]

    return state_space


if __name__ == "__main__":  # pragma: no cover
    """Run the main program with this function."""
    spring_const = 10.0
    mass = 0.5
    damping = 0.8

    ss = StateSpace(
        A=np.array([[0.0, 1.0], [-spring_const / mass, -damping / mass]]),
        B=np.array([[0.0], [1.0 / mass]]),
    )

    descritization_dt = 0.05
    ss = continuous_to_discrete(state_space=ss, dt=descritization_dt)
    ss.step_response(dt=descritization_dt, plot_response=True)
