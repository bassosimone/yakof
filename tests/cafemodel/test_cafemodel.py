"""Tests for the yakof.cafemodel package."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np

from yakof import cafemodel
from dt_model.engine.numpybackend import executor


def test_cafe_model():
    # Build model with standard inputs
    inputs = cafemodel.Inputs()
    model = cafemodel.build(inputs)

    # Set up test input values
    customers_sitin = np.array([10, 20, 5])  # Different numbers of sit-in customers
    customers_takeaway = np.array([15, 30, 10])  # Different numbers of takeaway customers

    # Get the correct enum values from the model
    weather_sunny_value = inputs.weather_sunny.value
    weather_rainy_value = inputs.weather_rainy.value
    time_morning_value = inputs.time_morning.value
    time_lunch_value = inputs.time_lunch.value
    time_afternoon_value = inputs.time_afternoon.value
    time_evening_value = inputs.time_evening.value

    # Weather and time scenarios (2 weather types × 4 time periods)
    weather_values = np.array(
        [
            weather_sunny_value,
            weather_sunny_value,
            weather_sunny_value,
            weather_sunny_value,
            weather_rainy_value,
            weather_rainy_value,
            weather_rainy_value,
            weather_rainy_value,
        ]
    )

    time_values = np.array(
        [
            time_morning_value,
            time_lunch_value,
            time_afternoon_value,
            time_evening_value,
            time_morning_value,
            time_lunch_value,
            time_afternoon_value,
            time_evening_value,
        ]
    )

    # Capacity values (can vary by scenario but using constants for test)
    indoor_seating = np.full(8, 10)
    outdoor_seating = np.full(8, 30)
    base_service_capacity = np.full(8, 4)
    base_takeaway_service_rate = np.full(8, 40)
    base_sitin_service_rate = np.full(8, 10)
    seat_turnover_rate = np.full(8, 1.4)

    # Set up state
    state = executor.State(
        values={
            inputs.customers_sitin.node: customers_sitin,
            inputs.customers_takeaway.node: customers_takeaway,
            inputs.weather_enum.tensor.node: weather_values,
            inputs.time_enum.tensor.node: time_values,
            inputs.indoor_seating.node: indoor_seating,
            inputs.outdoor_seating.node: outdoor_seating,
            inputs.base_service_capacity.node: base_service_capacity,
            inputs.base_takeaway_service_rate.node: base_takeaway_service_rate,
            inputs.base_sitin_service_rate.node: base_sitin_service_rate,
            inputs.seat_turnover_rate.node: seat_turnover_rate,
        }
    )

    # Evaluate the model
    for node in model.nodes:
        executor.evaluate(state, node)

    # Extract results
    seating_sustainable = state.values[model.seating_sustainable.node]
    service_sustainable = state.values[model.service_sustainable.node]
    seating_load = state.values[model.seating_load.node]
    service_load = state.values[model.service_load.node]

    # Basic shape checks
    assert seating_sustainable.shape == (
        3,
        1,
        8,
    ), "Seating sustainability has incorrect shape"
    assert service_sustainable.shape == (
        3,
        3,
        8,
    ), "Service sustainability has incorrect shape"

    # Check expected behaviors

    # 1. Sunny weather should have greater seating capacity than rainy
    #    (First 4 scenarios are sunny, last 4 are rainy)
    #    Check a sample customer case
    customer_idx = (0, 0)  # First sit-in, first takeaway

    # In sunny weather, sustainability should be better due to outdoor seating
    sunny_seating_sustainable = seating_sustainable[customer_idx[0], 0, 0]
    rainy_seating_sustainable = seating_sustainable[customer_idx[0], 0, 4]
    assert sunny_seating_sustainable or not rainy_seating_sustainable, (
        "Sunny weather should have better or equal seating sustainability than rainy"
    )

    # 2. Service capacity should be highest during lunch hours
    #    Check load during lunch hours (index 1 and 5) vs evening (index 3 and 7)
    lunch_service_load = service_load[customer_idx[0], customer_idx[1], 1]  # Lunch time index
    evening_service_load = service_load[customer_idx[0], customer_idx[1], 3]  # Evening time index
    assert lunch_service_load < evening_service_load, (
        "Lunch time should have lower service load due to increased capacity"
    )

    # 3. Verify general relationship between load and sustainability
    for i in range(seating_load.shape[0]):
        for j in range(seating_load.shape[1]):
            for k in range(seating_load.shape[2]):
                assert (seating_load[i, j, k] <= 1.0) == seating_sustainable[i, j, k], (
                    f"Seating sustainability at {i},{j},{k} doesn't match load threshold"
                )
                assert (service_load[i, j, k] <= 1.0) == service_sustainable[i, j, k], (
                    f"Service sustainability at {i},{j},{k} doesn't match load threshold"
                )
