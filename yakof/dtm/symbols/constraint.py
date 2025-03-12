from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ...frontend import graph


@runtime_checkable
class CumulativeDistribution(Protocol):
    """Protocol for classes allowing to sample from a cumulative distribution."""

    def cdf(self, x: float | np.ndarray, *args, **kwds) -> float | np.ndarray: ...


class Constraint:
    """
    Constraint class.

    This class is used to define constraints for the model.
    """

    def __init__(
        self,
        usage: graph.Node,
        capacity: graph.Node | CumulativeDistribution,
        group: str | None = None,
        name: str = "",
    ) -> None:
        self.usage = usage
        self.capacity = capacity
        self.name = name

        # TODO(bassosimone): this field is only used by the view. We could consider
        # deprecating it and moving the view mapping logic inside the view itself, which
        # would work as intended as long as we have a working __hash__. By doing this,
        # we would probably reduce the churn and coupling between the computational
        # model and the related view.
        self.group = group
