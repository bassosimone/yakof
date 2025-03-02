"""
Café model demonstrating phase space analysis with uncertainty.

This example shows how to:
1. Define a model with two presence variables (sit-in and takeaway customers)
2. Add probabilistic constraints (seating capacity follows a distribution)
3. Use conditional service rates based on load
4. Analyze the sustainability region with ensemble sampling
"""

import numpy as np

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

    # Use a fixed random seed to obtain reproducible output
    np.random.seed(4)

    # Prepare for building the model input arguments
    mab = minisimulator.ModelArgumentsBuilder()

    # Add the presence variables
    min_customers, max_customers, num_points = 0, 100, 10
    mab.add_linear_range(
        model_inputs.customers_sitin.node,
        minisimulator.LinearRange(
            start=min_customers,
            stop=max_customers,
            points=num_points,
        ),
    )
    mab.add_linear_range(
        model_inputs.customers_takeaway.node,
        minisimulator.LinearRange(
            start=min_customers,
            stop=max_customers,
            points=num_points,
        ),
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
    ensemble_size = 10000
    args = mab.build(ensemble_size)

    # --- Numerical Simulation ---

    # Create the simulation state
    state = executor.State(
        values=args,
        flags=graph.NODE_FLAG_TRACE,
    )

    # Evaluate each node in the linearized model
    for node in model_def.nodes:
        executor.evaluate(state, node)

    # --- Results ---

    # Extract, project, and print the seating_sustainable results
    seating_sustainable = state.values[model_def.seating_sustainable.node]
    seating_sustainable = seating_sustainable.sum(axis=2)
    seating_sustainable = seating_sustainable / ensemble_size
    print("=== Seating Sustainability ===")
    print(seating_sustainable)
    print("")

    # Extract, project, and print the service_sustainable results
    service_sustainable = state.values[model_def.service_sustainable.node]
    service_sustainable = service_sustainable.sum(axis=2)
    service_sustainable = service_sustainable / ensemble_size
    print("=== Service Sustainability ===")
    print(service_sustainable)
    print("")

    # Print the overall sustainability region
    sustainable = seating_sustainable * service_sustainable
    sustainable = sustainable > 0.5
    print("=== Sustainability Region ===")
    print(sustainable)
    print("")


if __name__ == "__main__":
    main()
