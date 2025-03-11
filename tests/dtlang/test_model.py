"""Tests for the yakof.dtlang.model module."""

# SPDX-License-Identifier: Apache-2.0

from dt_model import Ensemble

from yakof.dtlang import (
    Constraint,
    Index,
    Model,
    Piecewise,
    PresenceVariable,
    UniformCategoricalContextVariable,
)

from scipy import stats

import numpy as np


def test_model_evaluation_simple():
    """Evaluate the model is a relatively simple scenario and make
    sure the results we get are in line with the equivalent
    model as defined using the dt-model package."""

    # === Context Variables ===
    days = ["monday"]
    weekday = UniformCategoricalContextVariable("weekday", days)

    # === Presence Variables ===
    drink_customers = PresenceVariable(
        name="drink_customers",
        cvs=[weekday],
    )
    food_customers = PresenceVariable(
        "food_customers",
        cvs=[weekday],
    )

    # === Capacity Indexes ===
    seat_capacity = Index("seat_capacity", 10)

    service_capacity = Index("service_capacity", 20)

    # === Constraints ===
    seat_constraint = Constraint(
        usage=food_customers,
        capacity=seat_capacity,
        name="seat_constraint",
    )
    service_constraint = Constraint(
        usage=drink_customers + food_customers,
        capacity=service_capacity,
        name="service_constraint",
    )

    # === Model ===
    model = Model(
        name="restaurant",
        cvs=[weekday],
        pvs=[
            drink_customers,
            food_customers,
        ],
        indexes=[
            seat_capacity,
            service_capacity,
        ],
        capacities=[],
        constraints=[seat_constraint, service_constraint],
    )

    # === Evaluation ===
    grid = {
        drink_customers: np.linspace(0, 40, 5),
        food_customers: np.linspace(0, 40, 5),
    }

    ensemble = Ensemble(model, {weekday: days}) # type: ignore
    ccache: dict[Constraint, np.ndarray] = {}

    expect = np.array([
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])

    rv = model.evaluate(grid, ensemble)
    assert np.all(rv == expect)
