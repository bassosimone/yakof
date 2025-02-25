"""
Constraint
==========

The constraint expresses the relationship between a usage and
capacity variables to estimate the sustainability of the system.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import typing

from yakof.symbolic.symbol import Symbol


@typing.runtime_checkable
class Distribution(typing.Protocol):
    """Protocol modeling a scipy distribution."""

    def cdf(self, x: float | np.ndarray, *args, **kwds) -> float | np.ndarray: ...


class Constraint:
    """
    Constraint class.

    This class is used to define constraints for the model.
    """

    def __init__(
        self,
        usage: Symbol,
        capacity: Symbol | Distribution,
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
