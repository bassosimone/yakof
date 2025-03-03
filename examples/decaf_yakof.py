"""
..............
"""

from yakof.dtlang import (
    Constraint,
    Index,
    Model,
    PresenceVariable,
    UniformCategoricalContextVariable,
    piecewise,
    where,
)

# === Context Variables ===

values_weather = ["sunny", "cloudy", "rainy"]
weather = UniformCategoricalContextVariable("weather", values_weather)

values_time_of_day = ["morning", "lunch", "afternoon", "evening"]
time_of_day = UniformCategoricalContextVariable("time_of_day", values_time_of_day)

# === Presence Variables ===

customers_sitin = PresenceVariable(
    "customers_sitin",
    [
        weather,
        time_of_day,
    ],
)

customers_takeaway = PresenceVariable(
    "customers_takeaway",
    [
        weather,
        time_of_day,
    ],
)

# === Constant Indexes ===

seating_indoor_capacity = Index("seating_indoor_capacity", 10)
seating_outdoor_capacity = Index("seating_outdoor_capacity", 30)
base_service_capacity = Index("base_service_capacity", 4)
base_takeaway_service_rate = Index("base_takeaway_service_rate", 40)
base_sitin_service_rate = Index("base_sitin_service_rate", 10)
seat_turnover_rate = Index("seat_turnover_rate", 1.4)

# === Derived Indexes ===

seating_capacity = where(
    weather == "sunny",
    seating_indoor_capacity + seating_outdoor_capacity,
    seating_indoor_capacity,
)

service_capacity = piecewise(
    (
        (time_of_day == "morning", base_service_capacity * 2),
        (time_of_day == "lunch", base_service_capacity * 3),
        (time_of_day == "afternoon", base_service_capacity),
    ),
    default_value=base_service_capacity * 0.5,
)

efficiency_factor = piecewise(
    (
        (time_of_day == "morning", 1.2),
        (time_of_day == "lunch", 1.3),
        (time_of_day == "afternoon", 1.0),
    ),
    default_value=0.9,
)

takeaway_service_rate = base_takeaway_service_rate * efficiency_factor
sitin_service_rate = base_sitin_service_rate * efficiency_factor

# === Constraints ===

actual_seating_capacity = seating_capacity * seat_turnover_rate
seating = Constraint(usage=customers_sitin, capacity=actual_seating_capacity)

# Service constraint
servers_needed_sitin = customers_sitin / sitin_service_rate
servers_needed_takeaway = customers_takeaway / takeaway_service_rate
servers_needed_overall = servers_needed_sitin + servers_needed_takeaway
service = Constraint(usage=servers_needed_overall, capacity=service_capacity)

# === Model ===

model = Model(
    "café model",
    cvs=[weather, time_of_day],
    pvs=[customers_sitin, customers_takeaway],
    indexes=[
        seating_indoor_capacity,
        seating_outdoor_capacity,
        base_service_capacity,
        base_takeaway_service_rate,
        base_sitin_service_rate,
        seat_turnover_rate,
        seating_capacity,
        service_capacity,
        efficiency_factor,
        takeaway_service_rate,
        sitin_service_rate,
        actual_seating_capacity,
        servers_needed_sitin,
        servers_needed_takeaway,
        servers_needed_overall,
    ],
    capacities=[seating_indoor_capacity, seating_outdoor_capacity],
    constraints=[seating, service],
)

# === Numerical Evaluation ===

import numpy as np

from dt_model import Ensemble  # XXX get rid of this import

from yakof.frontend import graph

ensemble = Ensemble(model, {})  # XXX need to make the scenario WAI

grid = {
    customers_sitin: np.linspace(0, 100, 5),
    customers_takeaway: np.linspace(0, 100, 5),
}

field = model.evaluate(grid, ensemble, debugflags=graph.NODE_FLAG_TRACE)

print("=== Sustainability ===")
print(field)
