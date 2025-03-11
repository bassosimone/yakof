"""
A in the A/B testing
====================

This module contains the original implementation of model5, with minor
changes required to adapt it for A/B testing.

See https://github.com/maryamsajedi/coffee_dt/blob/master/coffee_dt_5.py.
"""

from dt_model import (
    UniformCategoricalContextVariable,
    PresenceVariable,
    Constraint,
    Ensemble,
    Model,
    Index,
)
from scipy.stats import triang

import numpy as np
from sympy import Symbol


def run() -> np.ndarray:
    """Runs the model and returns its result."""

    # === Context Variables ===
    days = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]
    CV_weekday = UniformCategoricalContextVariable("weekday", [Symbol(v) for v in days])

    # === Presence Variables ===
    drink_customers = PresenceVariable(
        name="drink_customers",
        cvs=[],
    )
    food_customers = PresenceVariable(
        "food_customers",
        cvs=[],
    )

    seat_capacity = Index("seat_capacity", 50)
    # Most common service capacity is 100, ranging from 60 to 100
    service_capacity = Index("service_capacity", triang(loc=80.0, scale=40.0, c=0.5))

    # Usage of drink customers from the seats available in the bar
    U_drink_seat = Index("drink customers seat usage factor", 0.2)

    # Usage of food customers from the seats available in the bar
    U_food_seat = Index("food customers seat usage factor", 0.8)

    # Usage of drink customers from the service
    U_service_drink = Index("drink customers service usage factor", 0.4)

    # Usage of food customers from the service
    U_service_food = Index("food customers service usage factor", 0.9)

    # === Constraints ===
    seat_constraint = Constraint(
        usage=drink_customers * U_drink_seat + food_customers * U_food_seat,  # type: ignore
        capacity=seat_capacity,
    )

    service_constraint = Constraint(
        usage=food_customers * U_service_food + drink_customers * U_service_drink,  # type: ignore
        capacity=service_capacity,
    )

    # === Model ===
    model = Model(
        "coffee_shop",
        cvs=[CV_weekday],
        pvs=[drink_customers, food_customers],
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

    # === Evaluation ===
    ensemble = Ensemble(model, {CV_weekday: days}, cv_ensemble_size=7)

    grid = {
        drink_customers: np.linspace(0, 100, 11),
        food_customers: np.linspace(0, 100, 11),
    }

    return model.evaluate(grid, ensemble)


if __name__ == "__main__":
    print(run())
