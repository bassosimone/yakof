from typing import Protocol, runtime_checkable

import numpy as np

from . import geometry


@runtime_checkable
class CapacityDistribution(Protocol):
    def cdf(self, x: float | np.ndarray, *args, **kwds) -> float | np.ndarray: ...


class Expression:

    def __init__(
        self,
        usage: geometry.ComputationTensor,
        capacity: geometry.ComputationTensor | CapacityDistribution,
    ) -> None:
        self.usage = usage
        self.capacity = capacity
