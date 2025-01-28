"""
Phase Space Analysis
===================

This module provides the fundamental abstractions for phase space analysis
of sustainability models, including support for ensemble analysis with
stochastic parameters.

The core concepts are:
- Range: A region of parameter space to explore
- Distribution: Statistical distribution for ensemble parameters
- Analysis: Configuration of a phase space analysis
- Result: Output from phase space analysis

Example:
    >>> analysis = phasespace.Analysis(
    ...     model=cafe_model,
    ...     parameters=(
    ...         ("customers_sitin", phasespace.Range(0, 100)),     # x-axis
    ...         ("customers_takeaway", phasespace.Range(0, 100)),  # y-axis
    ...     ),
    ...     conditions={"time": cafe_model.enums.time.morning},
    ...     observables=["sustainability"],
    ...     ensemble_parameters={
    ...         "service_capacity": phasespace.NormalDistribution(6, 1),
    ...     },
    ...     n_samples=100
    ... )
    >>> result = analysis.run()

SPDX-License-Identifier: Apache-2.0
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .. import backend
from . import model


@dataclass(frozen=True)
class Range:
    """Range in parameter space to explore.

    Args:
        start: Starting value of the range
        stop: Ending value of the range
        points: Number of points to sample (default: 100)
    """

    start: float
    stop: float
    points: int = 100

    def __post_init__(self):
        if self.stop <= self.start:
            raise ValueError("stop must be greater than start")
        if self.points < 2:
            raise ValueError("points must be at least 2")

    def linspace(self) -> np.ndarray:
        """Create linear space array for this range."""
        return np.linspace(self.start, self.stop, self.points)


class Distribution(Protocol):
    """Protocol for parameter distributions."""

    def sample(self, size: tuple[int, ...]) -> np.ndarray:
        """Sample from the distribution.

        Args:
            size: Shape of the output array

        Returns:
            Array of samples with given shape
        """
        ...


@dataclass
class NormalDistribution:
    """Normal (Gaussian) distribution."""

    mean: float
    std: float

    def sample(self, size: tuple[int, ...]) -> np.ndarray:
        """Generate normally distributed random samples."""
        return np.random.normal(self.mean, self.std, size=size)


@dataclass
class UniformDistribution:
    """Uniform distribution."""

    low: float
    high: float

    def sample(self, size: tuple[int, ...]) -> np.ndarray:
        """Generate uniformly distributed random samples."""
        return np.random.uniform(self.low, self.high, size=size)


@dataclass
class DiscreteDistribution(Distribution):
    """Distribution that samples from a discrete set of string values.

    Used for categorical parameters where values are drawn from a finite set
    of possibilities, like weather conditions or time periods.

    The distribution can be specified in two ways:
    1. As a sequence of strings for uniform sampling
    2. As a dict mapping strings to probabilities

    Args:
        values: Either a sequence of strings for uniform sampling,
               or a dict mapping strings to their probabilities.

    Example:
        >>> # Uniform sampling (all equally likely)
        >>> time = DiscreteDistribution(["morning", "afternoon", "evening"])
        >>> time.sample((2,))  # array(["morning", "evening"])

        >>> # Weighted sampling (with specified probabilities)
        >>> weather = DiscreteDistribution({
        ...     "sunny": 0.6,
        ...     "cloudy": 0.3,
        ...     "rainy": 0.1
        ... })
        >>> weather.sample((1,))  # array(["sunny"]) (with 0.6 probability)
    """

    values: Sequence[model.EnumValue] | Mapping[model.EnumValue, float]

    def __post_init__(self):
        if isinstance(self.values, dict):
            # Check probabilities sum to 1 within floating point tolerance
            if not np.isclose(sum(self.values.values()), 1.0):
                raise ValueError(
                    f"Probabilities must sum to 1.0, got {sum(self.values.values())}"
                )
            # Convert dict to parallel lists for numpy.choice
            self._choices = [v.get_value() for v in self.values.keys()]
            self._probabilities = list(self.values.values())
        else:
            # Uniform sampling
            self._choices = [v.get_value() for v in self.values]
            self._probabilities = None

    def sample(self, size: tuple[int, ...]) -> np.ndarray:
        """Sample from the discrete distribution."""
        return np.random.choice(
            self._choices, size=size, p=self._probabilities, replace=True
        )


@dataclass(frozen=True)
class Result:
    """Results from phase space analysis.

    Contains:
    - Parameter grids
    - Computed observables
    - Fixed conditions used
    - Ensemble information (if using ensemble analysis)
    """

    parameters: dict[str, np.ndarray]
    observables: dict[str, np.ndarray]
    conditions: dict[str, Any]
    ensemble_size: int = 1
    raw_samples: list[dict[str, np.ndarray]] | None = None

    def __str__(self) -> str:
        """Human readable representation."""
        params = ", ".join(
            f"{k}: [{v[0]:.1f}, {v[-1]:.1f}]" for k, v in self.parameters.items()
        )
        obs = ", ".join(self.observables.keys())
        conds = ", ".join(f"{k}: {v}" for k, v in self.conditions.items())
        ensemble = (
            f"\nEnsemble size: {self.ensemble_size}" if self.ensemble_size > 1 else ""
        )
        return (
            f"Phase Space Analysis Result\n"
            f"Parameters: {params}\n"
            f"Observables: {obs}\n"
            f"Conditions: {conds}"
            f"{ensemble}"
        )

    def std(self, observable: str) -> np.ndarray | None:
        """Compute standard deviation across ensemble."""
        if not self.raw_samples:
            return None
        return np.std([s[observable] for s in self.raw_samples], axis=0)

    def percentile(self, observable: str, q: float) -> np.ndarray | None:
        """Compute percentile across ensemble."""
        if not self.raw_samples:
            return None
        return np.percentile([s[observable] for s in self.raw_samples], q, axis=0)


@dataclass
class Analysis:
    """Configuration for phase space analysis.

    Args:
        model: The model to analyze
        parameters: Tuple of (name, range) pairs defining x and y axes
        conditions: Fixed conditions for the analysis
        observables: Names of tensors to compute
        ensemble_parameters: Optional stochastic parameters for ensemble analysis
        n_samples: Number of samples for ensemble analysis (default: 1)
    """

    model: model.Model
    parameters: tuple[tuple[str, Range], tuple[str, Range]]
    conditions: dict[str, Any]
    observables: list[str]
    ensemble_parameters: Mapping[str, Distribution] = field(default_factory=dict)
    n_samples: int = 1

    def __post_init__(self):
        if not self.observables:
            raise ValueError("observables cannot be empty")
        x_name, y_name = self.parameters[0][0], self.parameters[1][0]
        if x_name == y_name:
            raise ValueError("x and y parameter names must be different")
        if self.n_samples < 1:
            raise ValueError("n_samples must be >= 1")

    def run(self) -> Result:
        """Execute the analysis."""
        # Create parameter grids
        (x_name, x_range), (y_name, y_range) = self.parameters
        xx, yy = np.meshgrid(
            x_range.linspace(),
            y_range.linspace(),
            indexing="xy",  # Use cartesian indexing for intuitive plots
        )

        # For each point, run samples
        results = []
        for _ in range(self.n_samples):
            # Create bindings with current x, y values
            sample_bindings = self.conditions.copy()
            sample_bindings[x_name] = xx
            sample_bindings[y_name] = yy

            # Sample ensemble parameters if any
            for param, dist in self.ensemble_parameters.items():
                sample_bindings[param] = dist.sample(xx.shape)

            # Evaluate model
            ctx = backend.numpy_engine.PartialEvaluationContext(
                bindings=self.model.prepare_bindings(sample_bindings)
            )

            # Collect results for this sample
            sample_results = {}
            for name in self.observables:
                tensor = self.model.get_tensor_by_name(name)
                sample_results[name] = ctx.evaluate(tensor)
            results.append(sample_results)

        # Aggregate results (mean across samples)
        aggregated = {
            name: np.mean([r[name] for r in results], axis=0)
            for name in self.observables
        }

        return Result(
            parameters={
                x_name: x_range.linspace(),
                y_name: y_range.linspace(),
            },
            observables=aggregated,
            conditions=self.conditions,
            ensemble_size=self.n_samples,
            raw_samples=results,
        )
