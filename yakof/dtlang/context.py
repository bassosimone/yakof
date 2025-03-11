"""
Context Variables
=================

This module implements context variables (i.e., variables associated
with uncertainty) as graph.placeholder tensors in the XYZ space.

Two kind of context variables are supported:

1. categorical, where the variable assumes a discrete set of values
each associated with a given probability;

2. continuous, where the variable assumes a continuous range of values.

This implementation uses the original `dt-model` ensemble, thus, in
addition, we also need compatibility types to interact with it.

We compile both kind of context variables to graph.placeholder tensors
using, as for other types in this package, the XYZ space.
"""

from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import random

from . import geometry

from ..frontend import autoenum, graph


SampleWeight = float
"""The weight of a value sampled from an EnsembleSampler."""

SampleValue = float
"""SampledValue is the value sampled from an EnsembleSampler."""


@runtime_checkable
class EnsembleSampler(Protocol):
    """A protocol that generalizes over categorical and continuous distributions
    to provide a unified interface for sampling.

    Methods:
        support_size: Returns the size of the support of the distribution.
        sample: Samples weighted values from the distribution.
    """

    def support_size(self) -> int | None: ...

    def sample(
        self,
        nr: int = 1,
        *,
        subset: Sequence[str] | None = None,
        force_sample: bool = False,
    ) -> list[tuple[SampleWeight, SampleValue]]: ...


class CategoricalContextVariable(geometry.Tensor):
    """A context variable that can take on categorical values with
    associated probabilities.

    This class represents a categorical random variable that can be used
    in tensor computations.
    """

    def __init__(
        self,
        name: str,
        values: Mapping[str, float],
    ) -> None:
        """Initialize a categorical context variable.

        Args:
            name: The name of the context variable.
            values: A mapping from value names to their probabilities.
        """

        # 1. Ensure that we have at least one value in this context variable
        if len(values) <= 0:
            raise ValueError("values must be a non-empty sequence")

        # 2. Create the enumeration type and save the associated tensor
        self.__enum = autoenum.Type(geometry.space, name)
        super().__init__(geometry.space, self.__enum.tensor.node)

        # 3. Save the cateorial values and their probability
        self.__values = values

        # 4. Generate and save the enumeration IDs for each value
        self.__mapping = {v: autoenum.Value(self.__enum, v) for v in self.__values}

    # Redefine the lazy equality to allow for comparison with the original strings
    def __eq__(self, value: str) -> geometry.Tensor:  # type: ignore
        return geometry.space.equal(self.__enum.tensor, self.__mapping[value].tensor)

    # Redefine the identity hashing since we have redefined the equality
    def __hash__(self) -> int:
        return id(self)

    def support_size(self) -> int:
        """Returns the number of possible values in this categorical variable."""
        return len(self.__values)

    def sample(
        self,
        nr: int = 1,
        *,
        subset: Sequence[str] | None = None,
        force_sample: bool = False,
    ) -> list[tuple[SampleWeight, SampleValue]]:
        """Sample values from this categorical distribution.

        Args:
            nr: Number of samples to draw.
            subset: Optional subset of values to sample from.
            force_sample: Whether to force sampling even if nr >= support_size.

        Returns:
            A list of (probability, value) tuples.
        """
        # TODO: subset (if defined) should be a subset of the support (also: with repetitions?)

        keys, size = list(self.__values.keys()), self.support_size()
        if subset is not None:
            keys, size = subset, len(subset)

        if force_sample or nr < size:
            ratio = 1 / nr
            keys = random.choices(keys, k=nr)
        else:
            ratio = 1 / size

        return [(ratio, float(self.__mapping[k].value)) for k in keys]


# Ensure that the CategoricalContextVariable type implements EnsembleSampler
_: EnsembleSampler = CategoricalContextVariable("", {"a": 0.5, "b": 0.5})


class UniformCategoricalContextVariable(CategoricalContextVariable):
    """A categorical context variable where all values have equal probability."""

    def __init__(
        self,
        name: str,
        values: Sequence[str],
    ) -> None:
        """Initialize a uniform categorical context variable.

        Args:
            name: The name of the context variable.
            values: A sequence of possible values, all with equal probability.
        """
        # 1. Ensure that we have at least one value in this context variable
        if len(values) <= 0:
            raise ValueError("values must be a non-empty sequence")

        # 2. Defer to the parent class constructor
        super().__init__(name, {v: 1 / len(values) for v in values})


# Ensure that the UniformCategoricalContextVariable type implements EnsembleSampler
_: EnsembleSampler = UniformCategoricalContextVariable("", ["a", "b"])


ContextVariable = UniformCategoricalContextVariable | CategoricalContextVariable
"""A context variable is one of the many possible context variable types."""
