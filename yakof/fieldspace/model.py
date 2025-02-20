"""
Core Model Components
=====================

This module provides the fundamental abstractions for analyzing systems that
vary both in time and across an ensemble of possibilities. It builds upon
the orientation-aware tensor system from the backend to enforce strict
separation between different types of quantities.
"""

from __future__ import annotations
from typing import TypeVar, Any, Iterator

from yakof.backend import graph
from ..backend import oriented


# Axis constants
ENSEMBLE_AXIS = 0
TIME_AXIS = 1


# Define orientation types
class ScalarWise:
    """Orientation for scalar quantities."""


class TimeWise:
    """Orientation for time series data."""


class EnsembleWise:
    """Orientation for ensemble data."""


class FieldWise:
    """Orientation for `time x ensemble` data."""


def expand_time_to_field(
    tensor: oriented.Tensor[TimeWise],
) -> oriented.Tensor[FieldWise]:
    """Lift time tensor to field space by adding ensemble dimension."""
    return oriented.Tensor[FieldWise](graph.expand_dims(tensor.t, axis=ENSEMBLE_AXIS))


def expand_ensemble_to_field(
    tensor: oriented.Tensor[EnsembleWise],
) -> oriented.Tensor[FieldWise]:
    """Lift ensemble tensor to field space by adding time dimension."""
    return oriented.Tensor[FieldWise](graph.expand_dims(tensor.t, axis=TIME_AXIS))


def project_field_to_time_using_mean(
    tensor: oriented.Tensor[FieldWise],
) -> oriented.Tensor[TimeWise]:
    """Project field to time space by averaging over ensemble dimension."""
    return oriented.Tensor[TimeWise](graph.reduce_mean(tensor.t, axis=ENSEMBLE_AXIS))


def project_field_to_ensemble_using_mean(
    tensor: oriented.Tensor[FieldWise],
) -> oriented.Tensor[EnsembleWise]:
    """Project field to ensemble space by averaging over time dimension."""
    return oriented.Tensor[EnsembleWise](graph.reduce_mean(tensor.t, axis=TIME_AXIS))


def project_field_to_ensemble_using_sum(
    tensor: oriented.Tensor[FieldWise],
) -> oriented.Tensor[EnsembleWise]:
    """Project field to ensemble space by summing over time dimension."""
    return oriented.Tensor[EnsembleWise](graph.reduce_sum(tensor.t, axis=TIME_AXIS))


def project_time_to_scalar_using_sum(
    tensor: oriented.Tensor[TimeWise],
) -> oriented.Tensor[ScalarWise]:
    """Project time tensor to scalar space by summing."""
    return oriented.Tensor[ScalarWise](graph.reduce_sum(tensor.t))


def time_constant(value: float, name: str = "") -> oriented.Tensor[TimeWise]:
    """Create a constant in time space."""
    return oriented.Tensor[TimeWise](graph.constant(value))


def time_placeholder(
    name: str, default_value: float | None = None
) -> oriented.Tensor[TimeWise]:
    """Create a placeholder in time space."""
    return oriented.Tensor[TimeWise](
        graph.placeholder(name=name, default_value=default_value)
    )


def ensemble_constant(value: float, name: str = "") -> oriented.Tensor[EnsembleWise]:
    """Create a constant in ensemble space."""
    return oriented.Tensor[EnsembleWise](graph.constant(value))


def ensemble_placeholder(
    name: str, default_value: float | None = None
) -> oriented.Tensor[EnsembleWise]:
    """Create a placeholder in ensemble space."""
    return oriented.Tensor[EnsembleWise](
        graph.placeholder(name=name, default_value=default_value)
    )


def expand_scalar_to_field(
    tensor: oriented.Tensor[ScalarWise],
) -> oriented.Tensor[FieldWise]:
    """Lift scalar to field space by broadcasting to both ensemble and time dimensions."""
    return oriented.Tensor[FieldWise](
        graph.expand_dims(
            graph.expand_dims(tensor.t, axis=ENSEMBLE_AXIS), axis=TIME_AXIS
        )
    )


class Model:
    """
    Model combining time series and ensemble analysis.

    This class provides a convenient container for organizing analysis across
    different orientations. It maintains separate spaces for time-oriented,
    ensemble-oriented, and field-oriented quantities.
    """

    def __init__(self, name: str = ""):
        """
        Initialize model with separate spaces for different orientations.

        Args:
            name: Optional model name
        """
        self.time = oriented.TensorSpace[TimeWise](self)
        self.ensemble = oriented.TensorSpace[EnsembleWise](self)
        self.field = oriented.TensorSpace[FieldWise](self)
        self.scalar = oriented.TensorSpace[ScalarWise](self)
        self.raw_tensors: list[graph.Tensor] = []

    def append_tensor(self, tensor: graph.Tensor) -> None:
        """Append tensor to raw list for later evaluation."""
        self.raw_tensors.append(tensor)

    def iterable_graph(self) -> Iterator[graph.Tensor]:
        """Return iterable graph for evaluation."""
        return iter(self.raw_tensors)
