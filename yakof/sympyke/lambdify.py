"""
...
"""

from ..frontend import graph, linearize
from ..numpybackend import executor

import numpy as np


class lambdify:
    def __init__(self, target: graph.Node) -> None:
        self.target = target
        self.prog = linearize.forest(target)

    def __call__(self, state: executor.State, *args: np.ndarray) -> np.ndarray:
        for node in self.prog:
            executor.evaluate(state, node)
        return state.values[self.target]
