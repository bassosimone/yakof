"""
Statistical Models and Protocols
================================

This module defines the core data types and protocols for the statsx package.

It provides:
    - The Sample class for representing sampled values with weights
    - The Sampler protocol that defines the interface for all samplers
    - Type definitions for weights and values

This module serves as the foundation for both categorical and continuous
sampling implementations. Users should not typically import from this module
directly, but should instead use the top-level statsx package.
"""

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Generic, Protocol, Sequence, TypeVar, runtime_checkable

Probability = float
"""Probability value from a distribution, represents likelihood and should sum to 1."""

Weight = float
"""Weight associated with a sampled value, represents importance in an ensemble."""

Value = TypeVar("Value", float, int, contravariant=True)
"""Type variable for continuous (float) or categorical (int) values."""


@dataclass(frozen=True)
class Sample(Generic[Value]):
    """A sampled value with its associated weight.

    Attributes:
        weight: The weight of this sample in an ensemble (typically normalized to sum to 1).
        value: The sampled value (int for categorical, float for continuous).
    """

    weight: Weight
    value: Value


@runtime_checkable
class Sampler(Protocol, Generic[Value]):
    """Unified interface for sampling from distributions.

    This protocol allows consistent handling of both categorical and continuous
    distributions through a single interface.

    Methods:
        support_size():
            Returns the size of the distribution's support, for discrete
            distributions, or None, for continuous distributions.

        sample(count=1, *, subset=None, force_sample=False):
            Samples `count` values from the distribution.

            The implementation uses one of two sampling methods:
            - Random Variates Sampling: When count < support size, generating
              random samples with equal weights (1/count)
            - PDF-Based Sampling: When sampling the full support or a subset,
              returning each value with a weight proportional to its probability

            The choice between methods affects how the resulting samples should
            be interpreted and used in downstream applications.

            Args:
                count: Number of samples to draw
                subset: Optional subset of values to sample from
                force_sample: Forces random variates sampling even when
                    sampling the full support
    """

    def support_size(self) -> int | None: ...

    def sample(
        self,
        count: int = 1,
        *,
        subset: Sequence[Value] | None = None,
        force_sample: bool = False,
    ) -> Sequence[Sample]: ...
