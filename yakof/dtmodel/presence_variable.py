"""
Presence Variable
=================

A presence variable describes the amount of entities of a given
type within the digital twin system.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats

from ...symbolic.symbol import Symbol
from ...dtmodel.symbols.context_variable import ContextVariable


class PresenceVariable(Symbol):
    """
    Class to represent a presence variable.
    """

    def __init__(
        self,
        name: str,
        cvs: list[ContextVariable],
        distribution: Callable | None = None,
    ) -> None:
        super().__init__(name)
        self.cvs = cvs
        self.distribution = distribution

    def sample(
        self,
        cvs: dict | None = None,
        nr: int = 1,
    ) -> np.array:  # type: ignore
        """
        Returns a list of values sampled from the presence variable or provided
        subset.
        If a distribution is provided in the constructor, the values will be
        sampled according to that distribution.

        Parameters
        ----------
        cvs: dict
            Dict of context variables to sample.
        nr: int
            Number of values to sample.

        Returns
        -------
        np.array
            List of sampled values.
        """
        assert nr > 0

        all_cvs = []
        # TODO: check this functionality
        if cvs is not None:
            all_cvs = [cvs[cv] for cv in self.cvs if cv in cvs.keys()]
            # TODO: solve this issue of symbols vs names
            all_cvs = list(
                map(lambda v: v.name if isinstance(v, Symbol) else v, all_cvs)
            )
        assert self.distribution is not None  # TODO(bassosimone): double check this
        distr: dict = self.distribution(*all_cvs)
        return stats.truncnorm.rvs(
            -distr["mean"] / distr["std"],
            10,
            loc=distr["mean"],
            scale=distr["std"],
            size=nr,
        )
