import numpy as np
from sympy import Symbol

from yakof.dtmodel import (
    UniformCategoricalContextVariable,
    PresenceVariable,
    Constraint,
    Model,
    Index,
)


# Context Variables
days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
CV_weekday = UniformCategoricalContextVariable("weekday", [Symbol(v) for v in days])

# FIXME: we need to figure out the best way to pass
# placeholders, since this actually sucks!

# Presence Variables
drink_customers = PresenceVariable(
    name="drink_customers",
    cvs=[],
)
food_customers = PresenceVariable(
    "food_customers",
    cvs=[],
)

# Capacity Index
capacity = Index("capacity", 50)

# Usage Index
U_drink_customers = Index("drink service usage factor", 1)
U_food_customers = Index("food service usage factor", 1)

# Constraints
C_drink_customers = Constraint(
    usage=drink_customers * U_drink_customers + food_customers * U_food_customers,  # type: ignore
    capacity=capacity,
)

# Define the model
model = Model(
    "coffee_shop",
    cvs=[],
    pvs=[drink_customers, food_customers],
    indexes=[capacity, U_drink_customers, U_food_customers],
    constraints=[C_drink_customers],
    capacities=[capacity],
)
