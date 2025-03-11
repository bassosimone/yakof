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
import pytest


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

    ensemble = Ensemble(model, {weekday: days})  # type: ignore
    ccache: dict[Constraint, np.ndarray] = {}

    expect = np.array(
        [
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    rv = model.evaluate(grid, ensemble)
    assert np.all(rv == expect)


def test_model_initialization():
    """Test basic model initialization."""
    # Create minimal components
    weekday = UniformCategoricalContextVariable("weekday", ["monday"])
    x = PresenceVariable("x", [weekday])
    y = PresenceVariable("y", [weekday])
    idx = Index("idx", 10)

    # Create a simple constraint
    constraint = Constraint(x, idx, name="test_constraint")

    # Create model
    model = Model(
        name="test_model",
        cvs=[weekday],
        pvs=[x, y],
        indexes=[idx],
        capacities=[],
        constraints=[constraint],
    )

    # Check model properties
    assert model.name == "test_model"
    assert model.cvs == [weekday]
    assert model.pvs == [x, y]
    assert model.indexes == [idx]
    assert len(model.constraints) == 1
    assert model.constraints[0].name == "test_constraint"


def test_model_with_distribution_capacity():
    """Test model with probabilistic capacity distribution."""
    # Create components
    weekday = UniformCategoricalContextVariable("weekday", ["monday"])
    x = PresenceVariable("x", [weekday])
    y = PresenceVariable("y", [weekday])

    # Create a constraint with distribution capacity
    normal_dist = stats.norm(loc=10, scale=0.1)
    constraint = Constraint(x, normal_dist, name="prob_constraint")

    # Create model
    model = Model(
        name="prob_model",
        cvs=[weekday],
        pvs=[x, y],
        indexes=[],
        capacities=[],
        constraints=[constraint],
    )

    # Set up grid and ensemble
    grid = {x: np.array([9.0, 10.0, 11.0]), y: np.array([5.0, 10.0, 15.0])}

    ensemble = Ensemble(model, {weekday: ["monday"]})  # type: ignore

    # Evaluate model
    result = model.evaluate(grid, ensemble)

    # Verify shape
    assert result.shape == (3, 3)

    # Values should be probabilities from normal CDF
    # Since normal_dist has loc=10 and scale=0.1, evaluating at 9, 10, and 11:
    # - Values at x=9 should be close to 1.0 (1 - CDF(9) ≈ 1.0)
    # - Values at x=10 should be exactly 0.5 (1 - CDF(10) = 0.5)
    # - Values at x=11 should be close to 0.0 (1 - CDF(11) ≈ 0.0)

    # Decreasing values across the 1st row (as x increases)
    assert result[0, 0] - result[0, 1] > 1e-9
    assert result[0, 1] - result[0, 2] > 1e-9

    # Since all rows have same x distribution, verify first column is consistent
    assert abs(result[0, 0] - result[1, 0]) < 1e-9
    assert abs(result[1, 0] - result[2, 0]) < 1e-9


def test_model_with_more_than_two_presence_variables():
    """Test that model evaluation raises NotImplementedError when more than 2 presence variables are provided."""
    # Create context variable
    weekday = UniformCategoricalContextVariable("weekday", ["monday"])

    # Create three presence variables
    x = PresenceVariable("x", [weekday])
    y = PresenceVariable("y", [weekday])
    z = PresenceVariable("z", [weekday])

    # Create a simple constraint
    idx = Index("idx", 10)
    constraint = Constraint(x, idx)

    # Create model with three presence variables
    model = Model(
        name="3d_model",
        cvs=[weekday],
        pvs=[x, y, z],  # Three presence variables
        indexes=[idx],
        capacities=[],
        constraints=[constraint],
    )

    # Set up grid
    grid = {x: np.array([1, 2, 3]), y: np.array([4, 5, 6]), z: np.array([7, 8, 9])}

    ensemble = Ensemble(model, {weekday: ["monday"]})  # type: ignore

    # Should raise NotImplementedError due to 3 presence variables
    with pytest.raises(NotImplementedError, match="This model only supports 2D grids"):
        model.evaluate(grid, ensemble)
