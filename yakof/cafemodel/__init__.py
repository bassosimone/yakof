"""
Cafe Model.
==========

This module models cafe operations with capacity constraints across multiple dimensions:
- Sit-in customers: Occupying seating capacity
- Takeaway customers: Using service capacity without seating
- Ensemble: Representing different scenarios (weather, time of day) affecting operations

The model evaluates sustainability of cafe operations by checking:
1. Seating sustainability: Whether available seating can support sit-in customer demand
2. Service sustainability: Whether staff capacity can handle combined sit-in and takeaway demand

The model accounts for context-dependent factors:
- Weather: Affects available seating (outdoor seating only usable in sunny weather)
- Time of day: Affects service capacity and efficiency (e.g., more staff during lunch rush)
"""

from dataclasses import dataclass

from yakof.frontend import abstract, autoenum, autonaming, graph, linearize, morphisms

sitin_axis_id, takeaway_axis_id, ensemble_axis_id = morphisms.generate_canonical_axes(3)


class ScalarBasis:
    """Represents a scalar value without dimensions.

    Attributes
    ----------
        axes: Empty tuple indicating no dimensions
    """

    axes = ()


class SitinBasis:
    """Represents the sit-in customer dimension of the cafe model.

    Attributes
    ----------
        axes: Tuple containing the sit-in axis identifier
    """

    axes = (sitin_axis_id,)


class TakeawayBasis:
    """Represents the takeaway customer dimension of the cafe model.

    Attributes
    ----------
        axes: Tuple containing the takeaway axis identifier
    """

    axes = (takeaway_axis_id,)


class EnsembleBasis:
    """Represents different scenarios (weather, time) affecting cafe operations.

    Attributes
    ----------
        axes: Tuple containing the ensemble axis identifier
    """

    axes = (ensemble_axis_id,)


class WeatherBasis:
    """Represents weather conditions affecting cafe operations.

    Attributes
    ----------
        axes: Tuple containing the ensemble axis identifier
    """

    axes = (ensemble_axis_id,)


class TimeOfDayBasis:
    """Represents different times of day affecting cafe operations.

    Attributes
    ----------
        axes: Tuple containing the ensemble axis identifier
    """

    axes = (ensemble_axis_id,)


class FieldBasis:
    """Combined dimensions for the complete cafe modeling space.

    Attributes
    ----------
        axes: Tuple containing sit-in, takeaway and ensemble axis identifiers
    """

    axes = (sitin_axis_id, takeaway_axis_id, ensemble_axis_id)


sitin_space = abstract.TensorSpace(SitinBasis())

takeaway_space = abstract.TensorSpace(TakeawayBasis())

ensemble_space = abstract.TensorSpace(EnsembleBasis())

weather_space = abstract.TensorSpace(WeatherBasis())

time_of_day_space = abstract.TensorSpace(TimeOfDayBasis())

field_space = abstract.TensorSpace(FieldBasis())

expand_sitin_to_field = morphisms.ExpandDims(sitin_space, field_space)

expand_takeaway_to_field = morphisms.ExpandDims(takeaway_space, field_space)

expand_ensemble_to_field = morphisms.ExpandDims(ensemble_space, field_space)


class Inputs:
    """Input parameters for the cafe model.

    Attributes
    ----------
        customers_sitin: Number of sit-in customers across different scenarios
        customers_takeaway: Number of takeaway customers across different scenarios
        weather_enum: Enumeration type for weather conditions
        weather_sunny: Value representing sunny weather
        weather_rainy: Value representing rainy weather
        time_enum: Enumeration type for time of day
        time_morning: Value representing morning hours
        time_lunch: Value representing lunch hours
        time_afternoon: Value representing afternoon hours
        time_evening: Value representing evening hours
        indoor_seating: Number of indoor seats available
        outdoor_seating: Number of outdoor seats available (usable in sunny weather)
        base_service_capacity: Base number of staff members available
        base_takeaway_service_rate: Base number of takeaway customers that can be served per staff member
        base_sitin_service_rate: Base number of sit-in customers that can be served per staff member
        seat_turnover_rate: Rate at which seats become available for new customers
    """

    @autonaming.decorator
    def __init__(self):
        # Presence variables
        self.customers_sitin = sitin_space.placeholder("")
        self.customers_takeaway = takeaway_space.placeholder("")

        # Context: weather
        self.weather_enum = autoenum.Type(weather_space, "")
        self.weather_sunny = autoenum.Value(self.weather_enum, "")
        self.weather_rainy = autoenum.Value(self.weather_enum, "")

        # Context: time
        self.time_enum = autoenum.Type(time_of_day_space, "")
        self.time_morning = autoenum.Value(self.time_enum, "")
        self.time_lunch = autoenum.Value(self.time_enum, "")
        self.time_afternoon = autoenum.Value(self.time_enum, "")
        self.time_evening = autoenum.Value(self.time_enum, "")

        # Seating capacity
        self.indoor_seating = ensemble_space.placeholder("", 10)
        self.outdoor_seating = ensemble_space.placeholder("", 30)

        # Service capacity
        self.base_service_capacity = ensemble_space.placeholder("", 4)

        # Service rates
        self.base_takeaway_service_rate = ensemble_space.placeholder("", 40)
        self.base_sitin_service_rate = ensemble_space.placeholder("", 10)

        # Seating utilization factor
        self.seat_turnover_rate = ensemble_space.placeholder("", 1.4)


@dataclass(frozen=True)
class Model:
    """Ready to evaluate cafe model.

    Attributes
    ----------
        seating_sustainable: Boolean tensor indicating whether seating capacity
                            is sufficient across scenarios
        service_sustainable: Boolean tensor indicating whether service capacity
                            is sufficient across scenarios
        seating_load: Ratio of demand to capacity for seating
        service_load: Ratio of demand to capacity for service
        nodes: Ordered execution plan for evaluating the model graph
    """

    seating_sustainable: abstract.Tensor[FieldBasis]
    service_sustainable: abstract.Tensor[FieldBasis]
    seating_load: abstract.Tensor[FieldBasis]
    service_load: abstract.Tensor[FieldBasis]
    nodes: list[graph.Node]


def build(inputs: Inputs) -> Model:
    """Build a café sustainability model with uncertainty.

    This function constructs a model of cafe operations that evaluates sustainability
    across different scenarios, considering both seating and service constraints.

    The model accounts for:
    1. Weather conditions (sunny/rainy) affecting available seating
    2. Time of day affecting staff availability and efficiency
    3. Different service requirements for sit-in vs takeaway customers

    Args:
        inputs: Model input parameters and placeholders

    Returns
    -------
        Outputs: Model outputs including sustainability indicators
    """
    with autonaming.context():
        # --- context dependent variables ---

        # Compute the seating capacity based on the weather
        seating_capacity = ensemble_space.where(
            inputs.weather_enum.tensor == inputs.weather_sunny.tensor,
            inputs.indoor_seating + inputs.outdoor_seating,
            inputs.indoor_seating,
        )

        # Compute the service capacity based on the time of day
        service_capacity = ensemble_space.multi_clause_where(
            (
                (
                    inputs.time_enum.tensor == inputs.time_morning.tensor,
                    inputs.base_service_capacity * 2,
                ),
                (
                    inputs.time_enum.tensor == inputs.time_lunch.tensor,
                    inputs.base_service_capacity * 3,
                ),
                (
                    inputs.time_enum.tensor == inputs.time_afternoon.tensor,
                    inputs.base_service_capacity,
                ),
            ),
            default_value=inputs.base_service_capacity * 0.5,
        )

        # Service rates vary by time (servers work faster during rush hours)
        efficiency_factor = ensemble_space.multi_clause_where(
            (
                (
                    inputs.time_enum.tensor == inputs.time_morning.tensor,
                    ensemble_space.constant(1.2),
                ),
                (
                    inputs.time_enum.tensor == inputs.time_lunch.tensor,
                    ensemble_space.constant(1.3),
                ),
                (
                    inputs.time_enum.tensor == inputs.time_afternoon.tensor,
                    ensemble_space.constant(1.0),
                ),
            ),
            default_value=0.9,
        )
        takeaway_service_rate = inputs.base_takeaway_service_rate * efficiency_factor
        sitin_service_rate = inputs.base_sitin_service_rate * efficiency_factor

        # --- Space transformations ---

        # Expand inputs to the field space
        customers_sitin = expand_sitin_to_field(inputs.customers_sitin)
        customers_takeaway = expand_takeaway_to_field(inputs.customers_takeaway)
        seat_turnover_rate_field = expand_ensemble_to_field(inputs.seat_turnover_rate)

        # Expand intermediate results to the field space
        sitin_service_rate_field = expand_ensemble_to_field(sitin_service_rate)
        takeaway_service_rate_field = expand_ensemble_to_field(takeaway_service_rate)
        service_capacity_field = expand_ensemble_to_field(service_capacity)
        seating_capacity_field = expand_ensemble_to_field(seating_capacity)

        # --- Constraints ---

        # Seating constraint
        seating_load = customers_sitin / (seating_capacity_field * seat_turnover_rate_field)
        seating_sustainable = seating_load <= 1.0

        # Service constraint
        servers_needed_sitin = customers_sitin / sitin_service_rate_field
        servers_needed_takeaway = customers_takeaway / takeaway_service_rate_field
        servers_needed = servers_needed_sitin + servers_needed_takeaway
        service_load = servers_needed / service_capacity_field
        service_sustainable = service_load <= 1.0

    # Linearize the execution plan
    evaluation_plan = linearize.forest(
        seating_sustainable.node,
        service_sustainable.node,
        seating_load.node,
        service_load.node,
    )

    # --- Results ---
    return Model(
        seating_sustainable=seating_sustainable,
        service_sustainable=service_sustainable,
        seating_load=seating_load,
        service_load=service_load,
        nodes=evaluation_plan,
    )
