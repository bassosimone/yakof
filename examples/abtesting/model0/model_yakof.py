import numpy as np
from sympy import Symbol

from yakof.dtcompiler import (
    PresenceVariable,
    Constraint,
    Model,
    Index,
)

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
capacity = Index("capacity", value=50)

# Usage Index
U_drink_customers = Index("drink service usage factor", value=1)
U_food_customers = Index("food service usage factor", value=1)

# Constraints
C_drink_customers = Constraint(
    usage=Index(
        value=drink_customers.t * U_drink_customers.t
        + food_customers.t * U_food_customers.t
    ),
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
