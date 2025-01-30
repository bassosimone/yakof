"""
Field Space Analysis
===================

This package provides tools for analyzing systems that vary both in time and across
an ensemble of possibilities. It leverages the backend's orientation-aware tensor
system to:

1. Keep time-varying and ensemble-varying quantities strictly separated through
   compile-time type checking
2. Provide type-safe operations for combining and projecting fields
3. Support visualization of field distributions and statistics

The key abstraction is the orientation, of which we have four flavours:
- ScalarWise: tensor in the 0-dimensional space
- TimeWise: tensor in the 1-dimensional time space
- EnsembleWise: tensor in the 1-dimensional ensemble space
- FieldWise: tensor in the `time x ensemble` space

The orientation gets applied to an oriented.Tensor[O] or to an
oriented.TensorField[O] to produce tensors and fields.

Each oriented tensor type is managed in its corresponding oriented field
which ensures type safety through the orientation system.

Example:
    >>> model = fieldspace.Model()
    >>> # Define time-varying baseline
    >>> model.time.flow = fieldspace.time_constant([100, 120, 140])
    >>> # Define ensemble parameters with uncertainty
    >>> model.ensemble.sensitivity = fieldspace.ensemble_constant(np.random.normal(
    ...     size=(288,0),
    ...     loc=1.0,
    ...     scale=0.2
    ... ))
    >>> # Combine into field by lifting time tensor
    >>> field_flow = fieldspace.lift_time_to_field(model.time.flow)
    >>> # Apply ensemble variation
    >>> field_sensitivity = fieldspace.lift_ensemble_to_field(model.ensemble.sensitivity)
    >>> model.field.response = field_flow * field_sensitivity
    >>> # Visualize results
    >>> fieldspace.plot_distribution(model.field.response, model)

The package ensures that operations between differently oriented tensors are caught
at compile time, while providing convenient ways to lift tensors between spaces
when needed and to create tensors in the appropriate spaces.

SPDX-License-Identifier: Apache-2.0
"""

from .model import (
    Model,
    EnsembleWise,
    FieldWise,
    ScalarWise,
    TimeWise,
    ensemble_constant,
    ensemble_placeholder,
    expand_time_to_field,
    expand_ensemble_to_field,
    expand_scalar_to_field,
    project_field_to_time_using_mean,
    project_field_to_ensemble_using_mean,
    project_field_to_ensemble_using_sum,
    project_time_to_scalar_using_sum,
    time_constant,
    time_placeholder,
)
from .viz import plot_distribution, plot_confidence_intervals, plot_ensemble_paths

__all__ = [
    "Model",
    "EnsembleWise",
    "FieldWise",
    "ScalarWise",
    "TimeWise",
    "ensemble_constant",
    "ensemble_placeholder",
    "expand_time_to_field",
    "expand_ensemble_to_field",
    "expand_scalar_to_field",
    "plot_distribution",
    "plot_confidence_intervals",
    "project_field_to_time_using_mean",
    "project_field_to_ensemble_using_mean",
    "project_field_to_ensemble_using_sum",
    "project_time_to_scalar_using_sum",
    "plot_ensemble_paths",
    "time_constant",
    "time_placeholder",
]
