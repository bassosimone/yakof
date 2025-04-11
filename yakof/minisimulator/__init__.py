"""
Minimal Simulator.
=================

This package implements a minimal simulator for digital twins modeling. It provides
the essential functionality to define context variables for a given model and to
populate the ensemble dimension with appropriate values.

The simulator is intentionally "minimal" - it implements only the core functionality
needed to demonstrate integration capabilities with more complex systems without
reproducing the full complexity of the `dt-model` package. This focused approach
allows for a lightweight integration demonstration while maintaining extensibility
for more sophisticated implementations.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from dt_model.engine.frontend import graph


@dataclass(frozen=True)
class LinearRange:
    """Linear range to explore in a parameter space.

    Defines a one-dimensional range of values to be explored during simulation,
    creating evenly spaced points between the start and stop values.

    Args:
        start: Starting value of the range
        stop: Ending value of the range (must be greater than start)
        points: Number of points to sample (default: 100, minimum: 2)

    Raises
    ------
        ValueError: If stop <= start or if points < 2
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
        """Create linear space array for this range.

        Returns
        -------
            numpy.ndarray: Array of evenly spaced values from start to stop,
                          with the specified number of points.
        """
        return np.linspace(self.start, self.stop, self.points)


@runtime_checkable
class Distribution(Protocol):
    """Protocol for parameter distributions.

    Defines the interface for probability distributions used in ensemble simulations.
    Any class implementing this protocol can be used to generate random samples
    for parameter variation across ensemble members.

    Methods
    -------
        sample: Generate random samples from the distribution with the given shape.
        support_size: Return the number of distinct values in the distribution's support,
                     or None if the support is infinite or uncountable.
    """

    def sample(self, size: tuple[int, ...]) -> np.ndarray: ...

    def support_size(self) -> Optional[int]: ...


@dataclass(frozen=True)
class NormalDistribution:
    """Normal (Gaussian) distribution.

    Represents a normal probability distribution for ensemble parameter variation.

    Args:
        mean: Center of the distribution (expected value)
        std: Standard deviation (spread or width of the distribution)
    """

    mean: float
    std: float

    def sample(self, size: tuple[int, ...]) -> np.ndarray:
        """Generate samples from a normal distribution.

        Args:
            size: Shape of the output array

        Returns
        -------
            numpy.ndarray: Random samples with the specified shape, drawn
                          from a normal distribution with the given mean and std.
        """
        return np.random.normal(self.mean, self.std, size=size)

    def support_size(self) -> Optional[int]:
        """Return the size of the distribution's support.

        Returns
        -------
            None: Normal distributions have uncountable, infinite support
        """
        return None


@dataclass(frozen=True)
class UniformDistribution:
    """Uniform distribution.

    Represents a uniform probability distribution where all values in the
    specified range have equal probability of being sampled.

    Args:
        low: Lower boundary of the distribution range (inclusive)
        high: Upper boundary of the distribution range (exclusive)
    """

    low: float
    high: float

    def sample(self, size: tuple[int, ...]) -> np.ndarray:
        """Generate samples from a uniform distribution.

        Args:
            size: Shape of the output array

        Returns
        -------
            numpy.ndarray: Random samples with the specified shape, drawn
                          from a uniform distribution between low and high.
        """
        return np.random.uniform(self.low, self.high, size=size)

    def support_size(self) -> Optional[int]:
        """Return the size of the distribution's support.

        Returns
        -------
            None: Uniform continuous distributions have uncountable, infinite support
        """
        return None


@dataclass(frozen=True)
class DiscreteDistribution:
    """Discrete distribution.

    Represents a discrete probability distribution where samples are drawn from
    a finite set of possible values with specified probabilities.

    Args:
        choices: Sequence of possible values to sample from
        probabilities: Sequence of probabilities corresponding to each choice
                      (must sum to 1.0)
    """

    choices: Sequence[int]
    probabilities: Sequence[float]

    @staticmethod
    def with_uniform_probabilities(values: Sequence[int]) -> DiscreteDistribution:
        """Create a DiscreteDistribution with uniform probabilities.

        This factory method simplifies creation of discrete distributions where
        all values have equal probability of being sampled.

        Args:
            values: Sequence of possible values to sample from

        Returns
        -------
            DiscreteDistribution: A new discrete distribution with uniform probabilities
        """
        return DiscreteDistribution(
            choices=values,
            probabilities=[1.0 / len(values)] * len(values),
        )

    @staticmethod
    def with_discrete_probabilities(
        values: Sequence[tuple[int, float]],
    ) -> DiscreteDistribution:
        """Create a DiscreteDistribution with specified probabilities.

        This factory method simplifies creation of discrete distributions when values
        and their corresponding probabilities are stored as pairs.

        Args:
            values: Sequence of (value, probability) tuples

        Returns
        -------
            DiscreteDistribution: A new discrete distribution with values and probabilities
                                  extracted from the provided pairs
        """
        return DiscreteDistribution(
            choices=[value for value, _ in values],
            probabilities=[prob for _, prob in values],
        )

    def sample(self, size: tuple[int, ...]) -> np.ndarray:
        """Generate samples from a discrete distribution.

        Args:
            size: Shape of the output array

        Returns
        -------
            numpy.ndarray: Random samples with the specified shape, drawn
                          from the discrete distribution according to the
                          specified probabilities.
        """
        return np.random.choice(
            self.choices,
            size=size,
            p=self.probabilities,
            replace=True,
        )

    def support_size(self) -> int:
        """Return the size of the distribution's support.

        Returns
        -------
            int: The number of distinct possible values (choices) in this discrete distribution
        """
        return len(self.choices)


@dataclass(frozen=True)
class ConstantDistribution:
    """A distribution that always returns the same value.

    This represents a deterministic parameter in an otherwise
    stochastic ensemble - equivalent to a Dirac delta function.

    Args:
        value: The constant value to be returned by all samples

    """

    value: float | int | bool

    def sample(self, size: tuple[int, ...]) -> np.ndarray:
        """Generate an array filled with the constant value.

        Args:
            size: Shape of the output array

        Returns
        -------
            numpy.ndarray: Array of specified shape filled with the constant value
        """
        return np.full(size, self.value)

    def support_size(self) -> int:
        """Return the size of the distribution's support.

        Returns
        -------
            int: Always 1 for constant distributions, which have a single value
        """
        return 1


class ModelArgumentsBuilder:
    """Allows to programmatically build the model arguments.

    Attributes
    ----------
        params: A dictionary mapping nodes to distributions or linear ranges
    """

    def __init__(self):
        self.params: dict[graph.Node, Distribution | LinearRange] = {}

    def add(self, node: graph.Node, dlr: Distribution | LinearRange) -> None:
        """Add a distribution or linear range for a given node.

        Args:
            node: The node to associate with the distribution
            distr: The Distribution or LinearRange for generating values
        """
        self.params[node] = dlr

    def build(self, space_size: int) -> dict[graph.Node, np.ndarray]:
        """Build the model arguments by sampling values from the specified distributions.

        Args:
            space_size: The size of the ensemble space (number of samples to generate)

        Returns
        -------
            dict[graph.Node, np.ndarray]: A dictionary mapping nodes to arrays of sampled values
        """
        # 1. create an empty dictionary to store the results
        result: dict[graph.Node, np.ndarray] = {}

        # 2. insert the linear ranges and the distributions
        for node, lrange in self.params.items():
            if isinstance(lrange, LinearRange):
                result[node] = lrange.linspace()
                continue
            result[node] = lrange.sample((space_size,))

        # 3. return the results dictionary
        return result
