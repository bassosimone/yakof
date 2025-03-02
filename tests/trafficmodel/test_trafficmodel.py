"""Tests for the yakof.trafficmodel package."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from yakof.frontend import graph
from yakof.numpybackend import executor
from yakof import trafficmodel


def test_traffic_model():
    # Build model with standard inputs
    inputs = trafficmodel.Inputs()
    model = trafficmodel.build(inputs)

    # Set up test input values
    hours = np.array([6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    base_demand = np.array([10.0, 20.0, 50.0, 30.0, 15.0, 10.0, 10.0])
    price = np.array([1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    price_sensitivity = np.array([0.05, 0.1, 0.15])

    # Set up state
    state = executor.State(
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
    price_affected_demand = state.values[model.price_affected_demand.node]
    demand_after_removal = state.values[model.demand_after_removal.node]
    actual_demand = state.values[model.actual_demand.node]

    # Calculate expected results (simplified version for testing)
    # Peak hours are 7-9, so indices 1 and 2

    # Expected price effect calculation
    expected_price_effect = np.zeros((7, 3))
    for i in range(7):
        for j in range(3):
            expected_price_effect[i, j] = 1.0 - price_sensitivity[j] * np.log(
                price[i] / inputs.base_price
            )

    expected_price_affected = base_demand[:, np.newaxis] * expected_price_effect

    # Verify shape
    assert price_affected_demand.shape == (
        7,
        3,
    ), "Price affected demand has incorrect shape"

    # Check results match expected values (with tolerance for floating point)
    np.testing.assert_allclose(
        price_affected_demand, expected_price_affected, rtol=1e-5
    )

    # Verify that demand is reduced at peak and increased at shoulders
    # Peak hours: 7-9 (indices 1, 2)
    # Check that actual_demand is less than price_affected_demand in peak hours
    for i in [1, 2]:  # Peak hours
        for j in range(3):  # All ensemble dimensions
            assert (
                demand_after_removal[i, j] < price_affected_demand[i, j]
            ), f"Demand at peak hour {hours[i]} should be reduced"

    # Check total conservation of demand (within floating point tolerance)
    total_original = np.sum(price_affected_demand)
    total_final = np.sum(actual_demand)
    np.testing.assert_allclose(
        total_original,
        total_final,
        rtol=1e-5,
        err_msg="Total demand should be conserved after shifting",
    )

    # Additional checks for the time-shifting effect
    # Verify early shifting: Hour 6 should have increased demand
    assert np.all(
        actual_demand[0, :] > price_affected_demand[0, :]
    ), "Early shifting should increase demand in pre-peak hour"

    # Verify late shifting: Hour 9 should have increased demand
    assert np.all(
        actual_demand[3, :] > price_affected_demand[3, :]
    ), "Late shifting should increase demand in post-peak hour"
