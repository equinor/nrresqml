import dataclasses
from typing import TypedDict

import numpy as np


@dataclasses.dataclass
class ResQmlData:
    x0: float
    y0: float
    dx: float
    dy: float
    nx: int
    ny: int
    nz: int
    archel: np.ndarray
    cell_volumes: np.ndarray
    model_name: str


class BBox(TypedDict):
    x_min: float
    x_max: float
    y_min: float
    y_max: float
