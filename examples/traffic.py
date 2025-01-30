"""
Café Demand Model with Price and Time Effects
===========================================

This example demonstrates:
1. Price effects with ensemble variation in customer sensitivity
2. Time shifting of demand from peak to shoulder periods
3. Visualization using fieldspace plotting functions

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

from yakof import backend, fieldspace
from yakof.backend import numpy_engine

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
# Model Building
#


def build_combined_model() -> fieldspace.Model:
    """Build café model with both price effects and time shifting.

    This model:
    1. Applies price effects with ensemble variation
    2. Then applies time shifting to the price-affected demand
    """
    m = fieldspace.Model()

    # Time-varying inputs
    m.time.base_demand = fieldspace.time_placeholder("base_demand")
    m.time.price = fieldspace.time_placeholder("price")
    m.time.hours = fieldspace.time_placeholder("hours")

    # Ensemble-varying price sensitivity
    m.ensemble.price_sensitivity = fieldspace.ensemble_placeholder("price_sensitivity")

    # --- Time Space Calculations ---

    # Define time windows
    m.time.is_peak = (m.time.hours >= MORNING_PEAK_START) & (
        m.time.hours < MORNING_PEAK_END
    )

    m.time.is_early_window = (m.time.hours >= (MORNING_PEAK_START - 1.0)) & (
        m.time.hours < MORNING_PEAK_START
    )

    m.time.is_late_window = (m.time.hours >= MORNING_PEAK_END) & (
        m.time.hours < (MORNING_PEAK_END + 1.0)
    )

    # Count intervals in each window (for redistribution)
    m.scalar.early_window_size = fieldspace.project_time_to_scalar_using_sum(
        m.time.where(m.time.is_early_window, 1.0, 0.0)
    )
    m.scalar.late_window_size = fieldspace.project_time_to_scalar_using_sum(
        m.time.where(m.time.is_late_window, 1.0, 0.0)
    )

    # --- Field Space Calculations ---

    # Lift time series to field space
    m.field.base_demand = fieldspace.lift_time_to_field(m.time.base_demand)
    m.field.price = fieldspace.lift_time_to_field(m.time.price)
    m.field.is_peak = fieldspace.lift_time_to_field(m.time.is_peak)
    m.field.is_early_window = fieldspace.lift_time_to_field(m.time.is_early_window)
    m.field.is_late_window = fieldspace.lift_time_to_field(m.time.is_late_window)
    m.field.early_window_size = fieldspace.lift_scalar_to_field(
        m.scalar.early_window_size
    )
    m.field.late_window_size = fieldspace.lift_scalar_to_field(
        m.scalar.late_window_size
    )

    # Lift ensemble values to field space
    m.field.price_sensitivity = fieldspace.lift_ensemble_to_field(
        m.ensemble.price_sensitivity
    )

    # First apply price effects
    m.field.price_effect = 1.0 - m.field.price_sensitivity * m.field.log(
        m.field.price / BASE_PRICE
    )
    m.field.price_affected_demand = m.field.base_demand * m.field.price_effect

    # Then apply time shifting to the price-affected demand
    # Calculate peak demand and how much to remove
    m.field.peak_demand = m.field.where(
        m.field.is_peak, m.field.price_affected_demand, 0.0
    )

    # Remove demand from peak
    fraction_to_remove = EARLY_SHIFT_RATE + LATE_SHIFT_RATE
    m.field.demand_after_removal = m.field.where(
        m.field.is_peak,
        m.field.price_affected_demand * (1.0 - fraction_to_remove),
        m.field.price_affected_demand,
    )

    # Calculate total demand to shift to each shoulder
    m.field.total_early_shift = m.field.peak_demand * EARLY_SHIFT_RATE
    m.field.total_late_shift = m.field.peak_demand * LATE_SHIFT_RATE

    # Calculate total shift amounts
    m.ensemble.total_early = fieldspace.project_field_to_ensemble_using_sum(
        m.field.total_early_shift
    )
    m.ensemble.total_late = fieldspace.project_field_to_ensemble_using_sum(
        m.field.total_late_shift
    )

    # Distribute to shoulder periods
    m.field.total_early = fieldspace.lift_ensemble_to_field(m.ensemble.total_early)
    m.field.early_addition = m.field.where(
        m.field.is_early_window, m.field.total_early / m.field.early_window_size, 0.0
    )

    m.field.total_late = fieldspace.lift_ensemble_to_field(m.ensemble.total_late)
    m.field.late_addition = m.field.where(
        m.field.is_late_window, m.field.total_late / m.field.late_window_size, 0.0
    )

    # Final demand combines all effects
    m.field.actual_demand = (
        m.field.demand_after_removal + m.field.early_addition + m.field.late_addition
    )
    return m


#
# Visualization
#


def visualize_results(
    model: fieldspace.Model,
    ctx: backend.numpy_engine.PartialEvaluationContext,
    timestamps,
    plot_mask,
):
    """Create comprehensive visualization of café model results.

    Shows four demand patterns:
    1. Base demand
    2. Price-affected demand
    3. Demand after removal
    4. Final demand (with uncertainty)
    """
    fig, ((ax1, ax2, ax3), (ax4, ax5, _)) = plt.subplots(2, 3, figsize=(18, 12))

    # Get all demand components
    base_demand = ctx.evaluate(model.time.base_demand.t)
    price_affected = ctx.evaluate(model.field.price_affected_demand.t)
    demand_after_removal = ctx.evaluate(model.field.demand_after_removal.t)
    final_demand = ctx.evaluate(model.field.actual_demand.t)

    # Calculate y-axis limits to be shared
    all_demands = [
        base_demand[plot_mask],
        price_affected.mean(axis=0)[plot_mask],
        demand_after_removal.mean(axis=0)[plot_mask],
        final_demand.mean(axis=0)[plot_mask],
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
        timestamps[plot_mask], price_affected.mean(axis=0)[plot_mask], "b-", alpha=0.7
    )
    ax2.set_title("Price-Affected Demand")
    ax2.set_ylabel("Number of Customers")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(ylim)

    # Demand After Removal
    ax3.plot(
        timestamps[plot_mask],
        demand_after_removal.mean(axis=0)[plot_mask],
        "g-",
        alpha=0.7,
    )
    ax3.set_title("Demand After Removal")
    ax3.set_ylabel("Number of Customers")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(ylim)

    # Final Demand with uncertainty
    mean_demand = final_demand.mean(axis=0)
    std_demand = final_demand.std(axis=0)
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

    price = ctx.evaluate(model.time.price.t)
    ax5.plot(timestamps[plot_mask], price[plot_mask], "r-", linewidth=2)
    ax5.set_title("Price Pattern")
    ax5.set_ylabel("Price (€)")
    ax5.grid(True, alpha=0.3)

    # Format time axis
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    plt.tight_layout()


#
# Main Execution
#


def main():
    """Run combined effects model analysis."""
    # Setup time grid (15-minute intervals)
    intervals_per_hour = 4
    hours_to_model = 6
    n_intervals = hours_to_model * intervals_per_hour

    # Time points for x-axis
    timestamps = pd.date_range(start="06:00:00", periods=n_intervals, freq="15min")
    hours = CAFE_OPEN_HOUR + np.arange(n_intervals) / intervals_per_hour

    # Calculate plot limits
    plot_start = CAFE_OPEN_HOUR - PLOT_BUFFER
    plot_end = CAFE_CLOSE_HOUR + PLOT_BUFFER
    plot_mask = (hours >= plot_start) & (hours <= plot_end)

    # Generate base patterns
    base_demand = generate_base_demand(n_intervals)

    # Generate price pattern
    price = np.ones(n_intervals) * BASE_PRICE
    peak_mask = (hours >= MORNING_PEAK_START) & (hours <= MORNING_PEAK_END)
    price[peak_mask] = PEAK_PRICE

    # Generate ensemble price sensitivities
    price_sensitivity = np.random.normal(
        loc=PRICE_ELASTICITY,
        scale=PRICE_ELASTICITY_STD,
        size=(10,),
    )

    # Build and evaluate model
    model = build_combined_model()

    ctx = backend.numpy_engine.PartialEvaluationContext(
        bindings={
            "base_demand": base_demand,
            "hours": hours,
            "price": price,
            "price_sensitivity": price_sensitivity,
        },
        debug=True,
        cache=numpy_engine.DictCache(),
    )

    # Evaluate model
    ctx.evaluate_graph(model)

    # Visualize results
    visualize_results(model, ctx, timestamps, plot_mask)
    plt.show()

    # Print statistics
    print("\nDemand Statistics:")
    print(f"Total base demand: {base_demand.sum():.1f}")
    print(
        f"Mean price-affected demand: "
        f"{ctx.evaluate(model.field.price_affected_demand.t).mean(axis=0).sum():.1f}"
    )
    print(
        f"Mean final demand: "
        f"{ctx.evaluate(model.field.actual_demand.t).mean(axis=0).sum():.1f}"
    )
    print(
        f"Mean early shift: "
        f"{ctx.evaluate(model.field.early_addition.t).mean(axis=0).sum():.1f}"
    )
    print(
        f"Mean late shift: "
        f"{ctx.evaluate(model.field.late_addition.t).mean(axis=0).sum():.1f}"
    )

    price_sensitivity = ctx.evaluate(model.ensemble.price_sensitivity.t)
    print("\nPrice Sensitivity Statistics:")
    print(f"Mean: {price_sensitivity.mean():.3f}")
    print(f"Std Dev: {price_sensitivity.std():.3f}")


if __name__ == "__main__":
    main()
