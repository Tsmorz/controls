"""Add a doc string to my files."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StateSpaceData:
    """A state history object to store history of state information."""

    time: list[float] = field(default_factory=list)
    state: list[np.ndarray] = field(default_factory=list)
    control: list[np.ndarray] = field(default_factory=list)
    covariance: list[np.ndarray] = field(default_factory=list)

    def append_step(
        self, t: float, x: np.ndarray, cov: np.ndarray, u: np.ndarray
    ) -> None:
        self.time.append(t)
        self.state.append(x)
        self.control.append(u)
        self.covariance.append(cov)
