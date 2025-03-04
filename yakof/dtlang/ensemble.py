"""
Ensemble
========

Defines the types used by the `dt-model` ensemble package.
"""

from typing import Iterator

from .context import ContextVariable

Weight = float
"""The weight of an ensemble element."""

Variables = dict[ContextVariable, float]
"""Maps context variables to their values in the ensemble."""

Iter = Iterator[tuple[Weight, Variables]]
"""Iterator over the elements of the ensemble space."""
