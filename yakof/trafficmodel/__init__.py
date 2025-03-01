"""
Traffic Model
=============

This module models traffic demand patterns with price sensitivity and time-shifting effects.
It simulates how demand changes in response to pricing and how peak demand gets shifted to
shoulder periods (before and after peaks).

The model operates in multiple dimensions:
- Time: Represents different hours of the day
- Ensemble: Represents different scenarios or population segments with varying sensitivities
- Field: Combined time and ensemble dimensions for full modeling

The traffic model captures how demand is influenced by:
1. Price effects: Higher prices reduce demand based on price sensitivity (via formula: 1.0 - sensitivity * log(price/base_price))
2. Time-shifting: Peak-period demand partially shifts to shoulder periods while preserving total demand
"""

from dataclasses import dataclass

from ..frontend import abstract, autonaming, bases, graph, linearize, morphisms

# Axes
time_axis_id, ensemble_axis_id = morphisms.generate_canonical_axes(2)

# Bases
ScalarBasis = bases.Scalar


class TimeBasis:
    """Represents the time dimension in the traffic model.

    Attributes:
        axes: Tuple containing the time axis identifier
    """

    axes = (time_axis_id,)


class EnsembleBasis:
    """Represents different scenarios or population segments with varying sensitivities.

    Attributes:
        axes: Tuple containing the ensemble axis identifier
    """

    axes = (ensemble_axis_id,)


class FieldBasis:
    """Combined time and ensemble dimensions for the complete modeling space.

    Attributes:
        axes: Tuple containing both time and ensemble axis identifiers
    """

    axes = (time_axis_id, ensemble_axis_id)


# Spaces
scalar_space = abstract.TensorSpace(ScalarBasis())
time_space = abstract.TensorSpace(TimeBasis())
ensemble_space = abstract.TensorSpace(EnsembleBasis())
field_space = abstract.TensorSpace(FieldBasis())

# Scalar expansions
expand_scalar_to_time = morphisms.ExpandDims(scalar_space, time_space)
expand_scalar_to_ensemble = morphisms.ExpandDims(scalar_space, ensemble_space)
expand_scalar_to_field = morphisms.ExpandDims(scalar_space, field_space)

# Space expansions
expand_time_to_field = morphisms.ExpandDims(time_space, field_space)
expand_ensemble_to_field = morphisms.ExpandDims(ensemble_space, field_space)

# Field projections
project_field_to_time_using_sum = morphisms.ProjectUsingSum(field_space, time_space)
project_field_to_ensemble_using_sum = morphisms.ProjectUsingSum(
    field_space, ensemble_space
)
project_field_to_scalar_using_sum = morphisms.ProjectUsingSum(field_space, scalar_space)

# Time/Ensemble projections
project_time_to_scalar_using_sum = morphisms.ProjectUsingSum(time_space, scalar_space)
project_ensemble_to_scalar_using_sum = morphisms.ProjectUsingSum(
    ensemble_space, scalar_space
)


# Inputs
@dataclass(frozen=True)
class Inputs:
    """Input parameters for the traffic model.

    Attributes:
        morning_peak_start (float): Start time of the morning peak period (hour of day)
        morning_peak_end (float): End time of the morning peak period (hour of day)
        base_price (float): Reference price level used for price sensitivity calculations
        early_shift_rate (float): Fraction of peak demand that shifts to earlier periods
        late_shift_rate (float): Fraction of peak demand that shifts to later periods
        base_demand (abstract.Tensor[TimeBasis]): Baseline demand over time before any modifications
        price (abstract.Tensor[TimeBasis]): Time-varying price level
        hours (abstract.Tensor[TimeBasis]): Time points for the model (hours of day)
        price_sensitivity (abstract.Tensor[EnsembleBasis]): Sensitivity of different population segments to price changes
    """

    morning_peak_start = 7.0
    morning_peak_end = 9.0
    base_price = 1.0
    early_shift_rate = 0.3
    late_shift_rate = 0.1
    base_demand = time_space.placeholder("base_demand")
    price = time_space.placeholder("price")
    hours = time_space.placeholder("hours")
    price_sensitivity = ensemble_space.placeholder("price_sensitivity")


# Outputs
@dataclass(frozen=True)
class Outputs:
    """Output values from the traffic model.

    Attributes:
        price_affected_demand (abstract.Tensor[FieldBasis]): Demand after applying price sensitivity effects
                                                             following the formula: base_demand * (1 - sensitivity * log(price/base_price))
        demand_after_removal (abstract.Tensor[FieldBasis]): Demand after removing the shifted portion from peak periods
        actual_demand (abstract.Tensor[FieldBasis]): Final demand after all effects (price and time-shifting)
        nodes (list[graph.Node]): Ordered execution plan for evaluating the model graph
    """

    # Output tensors
    price_affected_demand: abstract.Tensor[FieldBasis]
    demand_after_removal: abstract.Tensor[FieldBasis]
    actual_demand: abstract.Tensor[FieldBasis]

    # Plan to evaluate the whole graph in order
    nodes: list[graph.Node]


def build(inputs: Inputs) -> Outputs:
    """Build the traffic model with price effects and time-shifting.

    This function constructs the full traffic model, applying both price sensitivity
    effects and time-shifting of peak demand to shoulder periods.

    Mathematical model:
    1. Price effect: price_effect = 1.0 - price_sensitivity * log(price/base_price)
    2. Price-affected demand: base_demand * price_effect
    3. Time-shifting: Remove (early_shift_rate + late_shift_rate) fraction of peak demand
       and redistribute to shoulder periods proportionally

    Args:
        inputs: Model input parameters and placeholders

    Returns:
        Outputs: Model outputs including the final demand pattern with all effects applied
    """
    # Auto-assign names to tensors to make debugging much easier
    with autonaming.context():

        # --- Time Space Calculations ---

        # Define time windows
        is_peak = (inputs.hours >= inputs.morning_peak_start) & (
            inputs.hours < inputs.morning_peak_end
        )
        is_early_distribution_window = (
            inputs.hours >= (inputs.morning_peak_start - 1.0)
        ) & (inputs.hours < inputs.morning_peak_end)
        is_late_distribution_window = (inputs.hours >= inputs.morning_peak_start) & (
            inputs.hours < (inputs.morning_peak_end + 1.0)
        )

        # Count intervals in each window (for redistribution)
        early_window_size = project_time_to_scalar_using_sum(
            time_space.where(
                is_early_distribution_window,
                time_space.constant(1.0),
                time_space.constant(0.0),
            )
        )
        late_window_size = project_time_to_scalar_using_sum(
            time_space.where(
                is_late_distribution_window,
                time_space.constant(1.0),
                time_space.constant(0.0),
            )
        )

        # --- Field Space Calculations ---

        # Lift time series to field space
        field_base_demand = expand_time_to_field(inputs.base_demand)
        field_price = expand_time_to_field(inputs.price)
        field_is_peak = expand_time_to_field(is_peak)
        field_is_early_window = expand_time_to_field(is_early_distribution_window)
        field_is_late_window = expand_time_to_field(is_late_distribution_window)
        field_early_window_size = expand_scalar_to_field(early_window_size)
        field_late_window_size = expand_scalar_to_field(late_window_size)

        # Lift ensemble values to field space
        field_price_sensitivity = expand_ensemble_to_field(inputs.price_sensitivity)

        # First apply price effects
        price_effect = 1.0 - field_price_sensitivity * field_space.log(
            field_price / inputs.base_price
        )
        price_affected_demand = field_base_demand * price_effect

        # Then apply time shifting to the price-affected demand
        # Calculate peak demand and how much to remove
        peak_demand = field_space.where(
            field_is_peak,
            price_affected_demand,
            field_space.constant(0.0),
        )

        # Remove demand from peak
        fraction_to_remove = inputs.early_shift_rate + inputs.late_shift_rate
        demand_after_removal = field_space.where(
            field_is_peak,
            price_affected_demand * (1.0 - fraction_to_remove),
            price_affected_demand,
        )

        # Calculate total demand to shift to each shoulder
        total_early_shift = peak_demand * inputs.early_shift_rate
        total_late_shift = peak_demand * inputs.late_shift_rate

        # Calculate total shift amounts
        total_early = project_field_to_ensemble_using_sum(total_early_shift)
        total_late = project_field_to_ensemble_using_sum(total_late_shift)

        # Distribute to shoulder periods
        field_total_early = expand_ensemble_to_field(total_early)
        early_addition = field_space.where(
            field_is_early_window,
            field_total_early / field_early_window_size,
            field_space.constant(0.0),
        )

        field_total_late = expand_ensemble_to_field(total_late)
        late_addition = field_space.where(
            field_is_late_window,
            field_total_late / field_late_window_size,
            field_space.constant(0.0),
        )

        # Final demand combines all the effects
        actual_demand = demand_after_removal + early_addition + late_addition

    # --- Output ---

    # Linearize the execution plan
    evaluation_plan = linearize.forest(
        price_affected_demand.node,
        demand_after_removal.node,
        actual_demand.node,
    )

    return Outputs(
        price_affected_demand=price_affected_demand,
        demand_after_removal=demand_after_removal,
        actual_demand=actual_demand,
        nodes=evaluation_plan,
    )
