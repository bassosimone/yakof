"""
Café model demonstrating phase space analysis with uncertainty.

This example shows how to:
1. Define a model with two presence variables (sit-in and takeaway customers)
2. Add probabilistic constraints (seating capacity follows a distribution)
3. Use conditional service rates based on load
4. Analyze the sustainability region with ensemble sampling
"""

from yakof.backend import graph
from yakof import phasespace

import matplotlib.pyplot as plt


def build_cafe_model():
    """Build a café sustainability model with uncertainty."""
    m = phasespace.Model("café model")

    # 1. Presence variables
    m.tensors.customers_sitin = m.placeholder(
        description="Number of sit-in customers", unit="customers/hour"
    )
    m.tensors.customers_takeaway = m.placeholder(
        description="Number of takeaway customers", unit="customers/hour"
    )

    # 2. Context variables
    # Weather impacts on seating!
    m.define_placeholder_enum("weather", "sunny", "rainy")

    # Time of day impact on the service
    m.define_placeholder_enum("time", "morning", "lunch", "afternoon", "evening")

    # Seating capacity (weather dependent)
    m.tensors.indoor_seating = m.placeholder(
        default_value=10,
        description="Number of indoor seats",
        unit="seats",
    )
    m.tensors.outdoor_seating = m.placeholder(
        default_value=30,
        description="Number of outdoor seats",
        unit="seats",
    )
    m.tensors.seating_capacity = graph.multi_clause_where(
        (
            m.tensors.weather == m.enums.weather.sunny,
            m.tensors.indoor_seating + m.tensors.outdoor_seating,
        ),
        (True, m.tensors.indoor_seating),
    )

    # Service capacity (time-dependent!)
    m.tensors.base_service_capacity = m.placeholder(
        default_value=4,
        description="Number of servers in the café",
        unit="servers",
    )
    m.tensors.service_capacity = graph.multi_clause_where(
        (m.tensors.time == m.enums.time.morning, m.tensors.base_service_capacity * 2),
        (m.tensors.time == m.enums.time.lunch, m.tensors.base_service_capacity * 3),
        (m.tensors.time == m.enums.time.afternoon, m.tensors.base_service_capacity),
        (True, m.tensors.base_service_capacity * 0.5),
    )

    # Service rates (also depend on whether we are at rush hours)
    m.tensors.base_takeaway_service_rate = m.placeholder(
        default_value=40,
        description="Number of takeaway customers a server can handle per hour",
        unit="customers/hour",
    )
    m.tensors.base_sitin_service_rate = m.placeholder(
        default_value=10,
        description="Number of sit-in customers a server can handle per hour",
        unit="customers/hour",
    )

    # Service rates vary by time (servers work faster during rush hours)
    m.tensors.efficiency_factor = graph.multi_clause_where(
        (m.tensors.time == m.enums.time.morning, 1.2),
        (m.tensors.time == m.enums.time.lunch, 1.3),
        (m.tensors.time == m.enums.time.afternoon, 1.0),
        (True, 0.9),
    )
    m.tensors.takeaway_service_rate = (
        m.tensors.base_takeaway_service_rate * m.tensors.efficiency_factor
    )
    m.tensors.sitin_service_rate = (
        m.tensors.base_sitin_service_rate * m.tensors.efficiency_factor
    )

    # Seating utilization factor
    m.tensors.seat_turnover_rate = m.placeholder(
        default_value=1.4,
        description="Number of customers that can be served per seat per hour",
        unit="customers/(seat * hour)",
    )

    # 3. Constraints
    # Seating constraint
    m.tensors.seating_load = m.tensors.customers_sitin / (
        m.tensors.seating_capacity * m.tensors.seat_turnover_rate
    )
    m.tensors.seating_sustainable = m.tensors.seating_load <= 1.0

    # Service constraint
    m.tensors.servers_needed_sitin = (
        m.tensors.customers_sitin / m.tensors.sitin_service_rate
    )
    m.tensors.servers_needed_takeaway = (
        m.tensors.customers_takeaway / m.tensors.takeaway_service_rate
    )
    m.tensors.servers_needed = (
        m.tensors.servers_needed_sitin + m.tensors.servers_needed_takeaway
    )
    m.tensors.service_load = m.tensors.servers_needed / m.tensors.service_capacity
    m.tensors.service_sustainable = m.tensors.service_load <= 1.0

    # 4. Overall metrics
    m.tensors.max_load = graph.maximum(m.tensors.seating_load, m.tensors.service_load)
    m.tensors.avg_load = (m.tensors.seating_load + m.tensors.service_load) / 2.0
    m.tensors.sustainability = (
        m.tensors.seating_sustainable * m.tensors.service_sustainable
    )
    return m


def main():
    """Run ensemble phase space analysis of café model."""
    model = build_cafe_model()

    # Create ensemble analysis
    observables = [
        # Seating-related
        "seating_load",
        "seating_sustainable",
        # Service-related
        "servers_needed_sitin",
        "servers_needed_takeaway",
        "servers_needed",
        "service_load",
        "service_sustainable",
        # Overall metrics
        "max_load",
        "avg_load",
        "sustainability",
    ]
    analysis = phasespace.Analysis(
        model=model,
        parameters=(
            ("customers_takeaway", phasespace.Range(0, 100)),
            ("customers_sitin", phasespace.Range(0, 100)),
        ),
        conditions={},  # No special conditions
        ensemble_parameters={
            "seat_turnover_rate": phasespace.NormalDistribution(1.4, 0.1),
            "sitin_service_rate": phasespace.NormalDistribution(10, 1),
            "takeaway_service_rate": phasespace.NormalDistribution(40, 4),
            "time": phasespace.DiscreteDistribution(
                (
                    model.enums.time.morning,
                    model.enums.time.lunch,
                    model.enums.time.afternoon,
                    model.enums.time.evening,
                )
            ),
            "weather": phasespace.DiscreteDistribution(
                {
                    model.enums.weather.sunny: 0.8,
                    model.enums.weather.rainy: 0.2,
                }
            ),
        },
        observables=observables,
        n_samples=1000,
    )

    # Run analysis
    result = analysis.run()

    # Create comparison plots organized by category
    fig, axs = plt.subplots(3, 3, figsize=(18, 15))

    # Add space between subplots and at the edges
    plt.subplots_adjust(
        left=0.1,  # Left margin
        right=0.95,  # Right margin
        bottom=0.1,  # Bottom margin
        top=0.9,  # Top margin
        wspace=0.15,  # Width spacing between subplots
        hspace=0.4,  # Height spacing between subplots
    )

    # Seating plots
    phasespace.plot_with_contours(
        result,
        observable="seating_load",
        ax=axs[0, 0],
        title="Seating Load",
        vmin=0,
        vmax=1,
        cmap="YlOrRd",
    )
    phasespace.plot(
        result,
        observable="seating_sustainable",
        ax=axs[0, 1],
        title="Seating Sustainable",
        cmap="RdYlGn",
    )

    # Service plots
    phasespace.plot(
        result,
        observable="servers_needed_sitin",
        ax=axs[0, 2],
        title="Servers Needed (Sit-in)",
        cmap="viridis",
    )
    phasespace.plot(
        result,
        observable="servers_needed_takeaway",
        ax=axs[1, 0],
        title="Servers Needed (Takeaway)",
        cmap="viridis",
    )
    phasespace.plot_with_contours(
        result,
        observable="service_load",
        ax=axs[1, 1],
        title="Service Load",
        vmin=0,
        vmax=1,
        cmap="YlOrRd",
    )
    phasespace.plot(
        result,
        observable="service_sustainable",
        ax=axs[1, 2],
        title="Service Sustainable",
        vmin=0,
        vmax=1,
        cmap="RdYlGn",
    )

    # Overall metrics
    phasespace.plot_with_contours(
        result,
        observable="max_load",
        ax=axs[2, 0],
        title="Maximum Load",
        vmin=0,
        vmax=1,
        cmap="YlOrRd",
    )
    phasespace.plot_with_contours(
        result,
        observable="avg_load",
        ax=axs[2, 1],
        title="Average Load",
        vmin=0,
        vmax=1,
        cmap="YlOrRd",
    )
    phasespace.plot(
        result,
        observable="sustainability",
        ax=axs[2, 2],
        title="Overall Sustainability",
        cmap="RdYlGn",
    )
    plt.suptitle("Café Sustainability Analysis")
    plt.show()


if __name__ == "__main__":
    main()
