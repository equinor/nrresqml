import dataclasses

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
