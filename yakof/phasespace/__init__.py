"""
Phase Space Analysis
===================

This package provides tools for exploring model behavior across parameter spaces.
Phase space analysis is a systematic approach to understanding how a model's
behavior changes as its parameters vary. It is particularly useful for:

1. Understanding model stability and sensitivity
2. Finding sustainable operating regions
3. Visualizing behavioral transitions
4. Monte Carlo exploration of uncertain parameters

Example:
    >>> from yakof import phasespace
    >>> analysis = phasespace.Analysis(
    ...     model=cafe_model,
    ...     parameters=(
    ...         ("customers_sitin", phasespace.Range(0, 100)),     # x-axis
    ...         ("customers_takeaway", phasespace.Range(0, 100)),  # y-axis
    ...     ),
    ...     conditions={"time": cafe_model.enums.time.morning},
    ...     observables=["sustainability"]
    ... )
    >>> result = analysis.run()
    >>> print(result)
    >>> # Optional visualization
    >>> phasespace.plot_result(result, title="Café Sustainability")

SPDX-License-Identifier: Apache-2.0
"""

from .analysis import (
    Analysis,
    DiscreteDistribution,
    Distribution,
    NormalDistribution,
    Range,
    Result,
    UniformDistribution,
)
from .model import Model
from .viz import plot, plot_with_contours

__all__ = [
    "Analysis",
    "DiscreteDistribution",
    "Distribution",
    "Model",
    "NormalDistribution",
    "Range",
    "Result",
    "UniformDistribution",
    "plot",
    "plot_with_contours",
]
