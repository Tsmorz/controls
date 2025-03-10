"""Add a doc string to my files."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class StateSpaceData:
    """A state history object to store history of state information."""

    time: np.ndarray
    state: list[np.ndarray] = field(default_factory=list)
    control: list[np.ndarray] = field(default_factory=list)
    covariance: Optional[list[np.ndarray]] = None
