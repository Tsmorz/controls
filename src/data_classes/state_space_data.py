"""Add a doc string to my files."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class StateSpaceData:
    """A state history object to store history of state information."""

    time: np.ndarray
    state: list[np.ndarray]
    control: Optional[list[np.ndarray]] = None
