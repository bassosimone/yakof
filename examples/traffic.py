"""
Café Demand Model with Price and Time Effects
===========================================

This example demonstrates:

1. Price effects with ensemble variation in customer sensitivity
2. Time shifting of demand from peak to shoulder periods

The model represents a café where:
- Base demand has a morning peak around 8:00 AM
- During peak hours (7:00-9:00 AM):
  * Price increases from €1.00 to €2.00
  * Customers have varying price sensitivity (ensemble)
  * Some customers shift to earlier/later times (30% early, 10% late)
"""

import numpy as np

from yakof import trafficmodel
from yakof.frontend import graph
from yakof.numpybackend import executor


def generate_base_demand(n_intervals, start_hour=6.0):
    """Generate synthetic café base demand with morning peak."""
    hours = start_hour + np.arange(n_intervals) / 4  # 4 intervals per hour
    demand = np.zeros(n_intervals)
    mask = (hours >= 6.0) & (hours < 21.0)  # Café open 6am-9pm
    open_hours = hours[mask]

    # Simple morning peak
    base = 5 + 80 * np.exp(-((open_hours - 8.0) ** 2) / 1.0)  # peak around 8:00
    demand[mask] = base
    return demand


def main():
    """Run café demand model analysis with tracing and console output."""
    # Use a fixed random seed to obtain reproducible output
    np.random.seed(4)

    # Setup time grid (15-minute intervals)
    intervals_per_hour = 4
    hours_to_model = 16  # 6am to 10pm
    n_intervals = hours_to_model * intervals_per_hour

    # Generate time points
    hours = 6.0 + np.arange(n_intervals) / intervals_per_hour

    # Generate base patterns
    base_demand = generate_base_demand(n_intervals)

    # Generate price pattern
    price = np.ones(n_intervals) * 1.0  # Base price €1.00
    peak_mask = (hours >= 7.0) & (hours <= 9.0)  # Peak hours 7-9am
    price[peak_mask] = 2.0  # Peak price €2.00

    # Generate ensemble price sensitivities
    ensemble_size = 10
    price_sensitivity = np.random.normal(
        loc=0.1,  # Mean price sensitivity
        scale=0.1,  # Standard deviation of price sensitivity
        size=(ensemble_size,),
    )

    # Create custom inputs with our parameters
    inputs = trafficmodel.Inputs(
        morning_peak_start=7.0,
        morning_peak_end=9.0,
        base_price=1.0,
        early_shift_rate=0.3,
        late_shift_rate=0.1,
    )

    # Build the traffic model
    model = trafficmodel.build(inputs)

    # Set up state with our input data
    state = executor.State(
        values={
            inputs.base_demand.node: base_demand,
            inputs.price.node: price,
            inputs.hours.node: hours,
            inputs.price_sensitivity.node: price_sensitivity,
        },
        flags=graph.NODE_FLAG_TRACE,  # Enable tracing
    )

    # Evaluate the model
    for node in model.nodes:
        executor.evaluate(state, node)

    # Extract results
    price_affected_demand = state.values[model.price_affected_demand.node]
    demand_after_removal = state.values[model.demand_after_removal.node]
    actual_demand = state.values[model.actual_demand.node]

    # Collapse the ensemble dimension
    actual_demand = np.mean(actual_demand, axis=1)

    # Print results to console
    print("=== Café Demand Model Results ===")
    print("\n--- Base Demand ---")
    print(base_demand)
    print("\n--- Actual Demand ---")
    print(actual_demand)


if __name__ == "__main__":
    main()
