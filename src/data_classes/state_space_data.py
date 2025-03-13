"""Add a doc string to my files."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class StateSpaceData:
    """A state history object to store history of state information."""

    time: list[float] = field(default_factory=list)
    state: list[np.ndarray] = field(default_factory=list)
    state_true: list[np.ndarray] = field(default_factory=list)
    control: list[np.ndarray] = field(default_factory=list)
    covariance: list[np.ndarray] = field(default_factory=list)

    def append_step(
        self,
        t: float,
        x: np.ndarray,
        x_truth: Optional[np.ndarray] = None,
        cov: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
    ) -> None:
        self.time.append(t)
        self.state.append(x)
        if u is not None:
            self.control.append(u)
        if cov is not None:
            self.covariance.append(cov)
        if x_truth is not None:
            self.state_true.append(x_truth)
