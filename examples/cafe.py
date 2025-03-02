"""
Café model demonstrating phase space analysis with uncertainty.

This example shows how to:
1. Define a model with two presence variables (sit-in and takeaway customers)
2. Add probabilistic constraints (seating capacity follows a distribution)
3. Use conditional service rates based on load
4. Analyze the sustainability region with ensemble sampling
"""

from yakof import cafemodel, minisimulator
from yakof.frontend import graph
from yakof.numpybackend import executor


def main():
    """Run ensemble phase space analysis of café model."""

    # --- Symbolic Model Setup ---

    # Create the model with default inputs
    model_inputs = cafemodel.Inputs()

    # Build café model definition
    model_def = cafemodel.build(model_inputs)

    # --- Numerical Setup ---

    # Prepare for building the model input arguments
    mab = minisimulator.ModelArgumentsBuilder()

    # Add the presence variables
    mab.add_linear_range(
        model_inputs.customers_sitin.node,
        minisimulator.LinearRange(start=0, stop=100, points=5),
    )
    mab.add_linear_range(
        model_inputs.customers_takeaway.node,
        minisimulator.LinearRange(start=0, stop=100, points=5),
    )

    # Add the base service capacity
    mab.add_distribution(
        model_inputs.base_service_capacity.node,
        minisimulator.NormalDistribution(mean=4, std=1),
    )

    # Add the seat turnover rate
    mab.add_distribution(
        model_inputs.seat_turnover_rate.node,
        minisimulator.NormalDistribution(mean=1.4, std=0.1),
    )

    # Add the base sitin service rate
    mab.add_distribution(
        model_inputs.base_sitin_service_rate.node,
        minisimulator.NormalDistribution(mean=10, std=1),
    )

    # Add the base takeaway service rate
    mab.add_distribution(
        model_inputs.base_takeaway_service_rate.node,
        minisimulator.NormalDistribution(mean=40, std=4),
    )

    # Add the time discrete distribution
    mab.add_distribution(
        model_inputs.time_enum.tensor.node,
        minisimulator.DiscreteDistribution.with_uniform_probabilities(
            [
                model_inputs.time_morning.value,
                model_inputs.time_lunch.value,
                model_inputs.time_afternoon.value,
                model_inputs.time_evening.value,
            ],
        ),
    )

    # Add the weather discrete distribution
    mab.add_distribution(
        model_inputs.weather_enum.tensor.node,
        minisimulator.DiscreteDistribution.with_discrete_probabilities(
            [
                (model_inputs.weather_sunny.value, 0.8),
                (model_inputs.weather_rainy.value, 0.2),
            ]
        ),
    )

    # Create the model arguments
    args = mab.build(4)

    # --- Numerical Simulation ---

    # Create the simulation state
    state = executor.State(
        values=args,
        flags=graph.NODE_FLAG_TRACE,
    )

    # Evaluate each node in the linearized model
    for node in model_def.nodes:
        executor.evaluate(state, node)


if __name__ == "__main__":
    main()
