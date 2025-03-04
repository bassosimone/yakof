from yakof.dtlang import (
    UniformCategoricalContextVariable,
    PresenceVariable,
    Constraint,
    Model,
    Index,
)
import numpy as np
from sympy import Symbol

from scipy.stats import triang


# Context Variables
days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
CV_weekday = UniformCategoricalContextVariable("weekday", days)

# Presence Variables
drink_customers = PresenceVariable(
    name="drink_customers",
    cvs=[],
)
food_customers = PresenceVariable(
    "food_customers",
    cvs=[],
)

# Capacity Index can be defined as a distribution, but I assumed seat capacity is fixed all the time.
seat_capacity = Index("seat_capacity", 50)
service_capacity = Index(
    "service_capacity", triang(loc=80.0, scale=40.0, c=0.5)
)  # Most common service capacity is 100, ranging from 60 to 100

U_drink_seat = Index(
    "drink customers seat usage factor", 0.2
)  # Usage of drink customers from the seats available in the bar
U_food_seat = Index(
    "food customers seat usage factor", 0.8
)  # Usage of food customers from the seats available in the bar

U_service_drink = Index(
    "drink customers service usage factor", 0.4
)  # Usage of drink customers from the service
U_service_food = Index(
    "food customers service usage factor", 0.9
)  # Usage of food customers from the service


# Constraints
seat_constraint = Constraint(
    usage=drink_customers * U_drink_seat + food_customers * U_food_seat,
    capacity=seat_capacity,
)

service_constraint = Constraint(
    usage=food_customers * U_service_food + drink_customers * U_service_drink,
    capacity=service_capacity,
)

# Define the model
model = Model(
    "coffee_shop",
    cvs=[CV_weekday],
    pvs=[
        drink_customers,
        food_customers,
    ],
    indexes=[
        seat_capacity,
        U_drink_seat,
        U_food_seat,
        U_service_drink,
        U_service_food,
        service_capacity,
    ],
    constraints=[seat_constraint, service_constraint],
    capacities=[seat_capacity, service_capacity],
)
