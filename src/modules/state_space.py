"""Basic docstring for my module."""

from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from jax import grad
from loguru import logger
from scipy.signal import cont2discrete

from config.definitions import FIG_SIZE, LEGEND_LOC, PLOT_ALPHA, PLOT_MARKER_SIZE
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

        if plot_response:  # pragma: no cover
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

        if plot_response:  # pragma: no cover
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

    @staticmethod
    def _add_bounds(ax, data, sigma, history, state, color) -> None:
        lower = data - 2.576 * sigma
        upper = data + 2.576 * sigma
        ax.fill_between(
            history.time,
            lower,
            upper,
            color=color,
            alpha=0.5,
            label=f"$x_{state} 99 C.I. $",
        )

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
            for num_state in range(num_states):
                data = np.array([arr[num_state] for arr in history.state]).flatten()

                sigma = np.array(
                    [arr[num_state, num_state] for arr in history.covariance]
                )
                if ii == 1:
                    data /= np.amax(np.abs(data))

                p = ax.plot(
                    history.time, data, "--", label=f"$x_{num_state}$", alpha=PLOT_ALPHA
                )
                self._add_bounds(
                    ax,
                    data=data,
                    sigma=sigma.flatten(),
                    history=history,
                    state=num_state,
                    color=p[0].get_color(),
                )
                ax.set_ylabel("State" if ii == 0 else "Normalized State")

                # plot the ground truth if it exists
                if len(history.state_true) > 0:
                    c = p[0].get_color()
                    data = np.array(
                        [arr[num_state] for arr in history.state_true]
                    ).flatten()
                    ax.plot(
                        history.time,
                        data,
                        ".",
                        label=f"$x_{num_state} (true)$",
                        alpha=PLOT_ALPHA,
                        color=c,
                        markersize=PLOT_MARKER_SIZE,
                    )
            for u in range(history.control[0].shape[1]):
                control = [arr[u] for arr in history.control]
                if ii == 1:
                    control /= np.amax(np.abs(control))
                ax.step(history.time, control, label="control input", alpha=PLOT_ALPHA)

        for ax in axs:
            ax.set_xlabel("Time (s)")
            ax.grid(True)
            ax.legend(loc=LEGEND_LOC)
            ax.set_xlim(min(history.time) - 0.5, max(history.time) + 0.5)

        plt.show()
        plt.close()

    @staticmethod
    def plot_states(
        history: StateSpaceData, title: str = "State Space History"
    ) -> None:  # pragma: no cover
        """Plot the history of state space model.

        :param history: State history object
        :param title: Plot title
        :return: None
        """
        fig, axs = plt.subplots(1, 1, sharex=True, figsize=FIG_SIZE)
        plt.suptitle(title)

        state_0 = np.array([arr[0] for arr in history.state])
        state_1 = np.array([arr[1] for arr in history.state])
        state_0, state_1 = state_0.flatten(), state_1.flatten()

        axs.scatter(state_0, state_1, s=PLOT_MARKER_SIZE, label="$ x_1 vs x_2$")

        axs.grid(True)
        axs.set_aspect("equal")
        axs.legend(loc=LEGEND_LOC)
        axs.set_xlabel("$x_0$")
        axs.set_ylabel("$x_1$")

        plt.show()
        plt.close()

    def continuous_to_discrete(self, discretization_dt) -> None:
        """Convert a continuous state space to discrete state space."""
        system = (self.A, self.B, self.C, self.D)
        system_disc = cont2discrete(system, dt=discretization_dt, method="zoh")

        self.A = system_disc[0]
        self.B = system_disc[1]
        self.C = system_disc[2]
        self.D = system_disc[3]


def mass_spring_damper_model(
    mass: float = 0.5,
    spring_const: float = 20.0,
    damping: float = 0.4,
    discretization_dt: float = 0.0,
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

    def linearize(self, x: np.ndarray) -> StateSpace:
        """Linearize a list of callables.

        :return: Jacobian matrix
        """
        jacobian = np.zeros((len(self.f), len(x)))
        for ii, f in enumerate(self.f):
            grad_f = grad(f, argnums=(0, 1, 2))
            jacobian[ii, :] = grad_f(x[0, 0], x[1, 0], x[2, 0])
        return StateSpace(A=jacobian, B=jacobian)

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Step the state-space model by one step.

        :param x: Current state
        :param u: Control input
        :return: Next state
        """
        ss = self.linearize(x=x)
        return ss.A @ x + ss.B @ u


if __name__ == "__main__":  # pragma: no cover
    """Run the main program with this function."""
    dt = 0.05
    spd_ss = mass_spring_damper_model(discretization_dt=dt)
    spd_ss.step_response(delta_t=dt, plot_response=True)
    spd_ss.impulse_response(delta_t=dt, plot_response=True)
