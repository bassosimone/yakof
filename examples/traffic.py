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
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from yakof import trafficmodel
from yakof.frontend import abstract, graph
from yakof.numpybackend import executor

#
# Constants
#

# Business operation hours
CAFE_OPEN_HOUR = 6.0  # 6:00 AM
CAFE_CLOSE_HOUR = 21.0  # 9:00 PM
PLOT_BUFFER = 0.5  # Show 30' before and after opening hours

# Peak time window (7:00-9:00 AM)
MORNING_PEAK_START = 7.0
MORNING_PEAK_END = 9.0

# Price parameters
BASE_PRICE = 1.0
PEAK_PRICE = 2.0
PRICE_ELASTICITY = 0.1  # Mean price sensitivity
PRICE_ELASTICITY_STD = 0.1  # Standard deviation of price sensitivity

# Time shifting parameters
EARLY_SHIFT_RATE = 0.3  # 30% shifts earlier
LATE_SHIFT_RATE = 0.1  # 10% shifts later

#
# Data Generation
#


def generate_base_demand(
    n_intervals: int, start_hour: float = CAFE_OPEN_HOUR
) -> np.ndarray:
    """Generate synthetic café base demand with morning peak.

    Args:
        n_intervals: Number of 15-minute intervals
        start_hour: Starting hour for the simulation
    """
    hours = start_hour + np.arange(n_intervals) / 4  # 4 intervals per hour
    demand = np.zeros(n_intervals)
    mask = (hours >= CAFE_OPEN_HOUR) & (hours < CAFE_CLOSE_HOUR)
    open_hours = hours[mask]

    # Simple morning peak
    base = 5 + 80 * np.exp(-((open_hours - 8.0) ** 2) / 1.0)  # peak around 8:00
    demand[mask] = base
    return demand


#
# Visualization
#


def saveviz(
    price_affected_demand: np.ndarray,
    demand_after_removal: np.ndarray,
    actual_demand: np.ndarray,
    base_demand: np.ndarray,
    price: np.ndarray,
    timestamps: pd.DatetimeIndex,
    plot_mask: np.ndarray,
    output_file: str,
) -> None:
    """Create comprehensive visualization of café model results.

    Shows four demand patterns:
    1. Base demand
    2. Price-affected demand
    3. Demand after removal
    4. Final demand (with uncertainty)

    Args:
        price_affected_demand: Demand after price sensitivity effects
        demand_after_removal: Demand after removing peak-period demand
        actual_demand: Final demand after all effects
        base_demand: Original demand before any effects
        price: Price at each time interval
        timestamps: Time points for x-axis
        plot_mask: Mask for filtering data to plot range
        output_file: File path to save the plot
    """
    fig, ((ax1, ax2, ax3), (ax4, ax5, _)) = plt.subplots(2, 3, figsize=(18, 12))

    # Calculate y-axis limits to be shared
    all_demands = [
        base_demand[plot_mask],
        price_affected_demand.mean(axis=1)[plot_mask],
        demand_after_removal.mean(axis=1)[plot_mask],
        actual_demand.mean(axis=1)[plot_mask],
    ]
    ymin = min(np.min(d) for d in all_demands)
    ymax = max(np.max(d) for d in all_demands)
    y_margin = 0.1 * (ymax - ymin)  # 10% margin
    ylim = (ymin - y_margin, ymax + y_margin)

    # Base Demand
    ax1.plot(timestamps[plot_mask], base_demand[plot_mask], "k-", alpha=0.7)
    ax1.set_title("Base Demand")
    ax1.set_ylabel("Number of Customers")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(ylim)

    # Price Affected Demand
    ax2.plot(
        timestamps[plot_mask],
        price_affected_demand.mean(axis=1)[plot_mask],
        "b-",
        alpha=0.7,
    )
    ax2.set_title("Price-Affected Demand")
    ax2.set_ylabel("Number of Customers")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(ylim)

    # Demand After Removal
    ax3.plot(
        timestamps[plot_mask],
        demand_after_removal.mean(axis=1)[plot_mask],
        "g-",
        alpha=0.7,
    )
    ax3.set_title("Demand After Removal")
    ax3.set_ylabel("Number of Customers")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(ylim)

    # Final Demand with uncertainty
    mean_demand = actual_demand.mean(axis=1)
    std_demand = actual_demand.std(axis=1)
    ax4.plot(timestamps[plot_mask], mean_demand[plot_mask], "r-", alpha=0.7)
    ax4.fill_between(
        timestamps[plot_mask],
        (mean_demand - std_demand)[plot_mask],
        (mean_demand + std_demand)[plot_mask],
        color="r",
        alpha=0.2,
    )
    ax4.set_title("Final Demand (with uncertainty)")
    ax4.set_ylabel("Number of Customers")
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(ylim)

    ax5.plot(timestamps[plot_mask], price[plot_mask], "r-", linewidth=2)
    ax5.set_title("Price Pattern")
    ax5.set_ylabel("Price (€)")
    ax5.grid(True, alpha=0.3)

    # Format time axis
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    plt.tight_layout()

    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved to {output_file}")


#
# Main Execution
#


def main() -> None:
    """Run combined effects model analysis."""
    # Setup time grid (15-minute intervals)
    intervals_per_hour: int = 4
    hours_to_model: int = 6
    n_intervals: int = hours_to_model * intervals_per_hour

    # Time points for x-axis
    timestamps: pd.DatetimeIndex = pd.date_range(
        start="06:00:00", periods=n_intervals, freq="15min"
    )
    hours: np.ndarray = CAFE_OPEN_HOUR + np.arange(n_intervals) / intervals_per_hour

    # Calculate plot limits
    plot_start: float = CAFE_OPEN_HOUR - PLOT_BUFFER
    plot_end: float = CAFE_CLOSE_HOUR + PLOT_BUFFER
    plot_mask: np.ndarray = (hours >= plot_start) & (hours <= plot_end)

    # Generate base patterns
    base_demand: np.ndarray = generate_base_demand(n_intervals)

    # Generate price pattern
    price: np.ndarray = np.ones(n_intervals) * BASE_PRICE
    peak_mask: np.ndarray = (hours >= MORNING_PEAK_START) & (hours <= MORNING_PEAK_END)
    price[peak_mask] = PEAK_PRICE

    # Generate ensemble price sensitivities
    price_sensitivity: np.ndarray = np.random.normal(
        loc=PRICE_ELASTICITY,
        scale=PRICE_ELASTICITY_STD,
        size=(10,),
    )

    # Create custom inputs with our parameters
    inputs: trafficmodel.Inputs = trafficmodel.Inputs(
        morning_peak_start=MORNING_PEAK_START,
        morning_peak_end=MORNING_PEAK_END,
        base_price=BASE_PRICE,
        early_shift_rate=EARLY_SHIFT_RATE,
        late_shift_rate=LATE_SHIFT_RATE,
    )

    # Build the traffic model
    model: trafficmodel.Outputs = trafficmodel.build(inputs)

    # Set up state with our input data
    state: executor.State = executor.State(
        values={
            inputs.base_demand.node: base_demand,
            inputs.price.node: price,
            inputs.hours.node: hours,
            inputs.price_sensitivity.node: price_sensitivity,
        }
    )

    # Evaluate the model
    for node in model.nodes:
        executor.evaluate(state, node)

    # Extract results
    price_affected_demand: np.ndarray = state.values[model.price_affected_demand.node]
    demand_after_removal: np.ndarray = state.values[model.demand_after_removal.node]
    actual_demand: np.ndarray = state.values[model.actual_demand.node]

    # Visualize results
    saveviz(
        price_affected_demand,
        demand_after_removal,
        actual_demand,
        base_demand,
        price,
        timestamps,
        plot_mask,
        "cafe_model.png",
    )

    # Print statistics
    print("\nDemand Statistics:")
    print(f"Total base demand: {base_demand.sum():.1f}")
    print(f"Mean price-affected demand: {price_affected_demand.mean(axis=1).sum():.1f}")
    print(f"Mean final demand: {actual_demand.mean(axis=1).sum():.1f}")

    print("\nPrice Sensitivity Statistics:")
    print(f"Mean: {price_sensitivity.mean():.3f}")
    print(f"Std Dev: {price_sensitivity.std():.3f}")


if __name__ == "__main__":
    main()
