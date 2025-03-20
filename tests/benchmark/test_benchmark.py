"""Tests for the yakof.benchmark package."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np

from yakof import benchmark


def test_benchmark_run():
    """Test the benchmark run function covers all code paths with minimal execution time."""

    # Simple test factory for quick execution
    def test_factory():
        def compute(x, y):
            return x + y

        return compute

    # Small arrays for fast testing
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)

    # Run with minimal iterations to keep test fast
    results = benchmark.run(x, y, test_factory, n_runs=2)

    # Basic sanity checks
    assert isinstance(results, benchmark.Results)
    assert len(results.creation_times) == 2
    assert len(results.compute_times) == 2
    assert results.creation_memory_peak > 0
    assert results.compute_memory_peak > 0

    # Test the property accessors
    assert results.mean_creation_time > 0
    assert results.mean_compute_time > 0
    assert results.std_creation_time >= 0
    assert results.std_compute_time >= 0

    # Check that the returned function works correctly
    func = test_factory()
    xx, yy = np.meshgrid(x, y)
    result = func(xx, yy)
    expected = xx + yy
    np.testing.assert_array_equal(result, expected)


def test_results_class():
    """Test the Results class properties with controlled values."""

    # Create a Results object with known values
    results = benchmark.Results()
    results.creation_times = [1.0, 2.0, 3.0]
    results.compute_times = [4.0, 5.0, 6.0]
    results.creation_memory_peak = 1000
    results.compute_memory_peak = 2000

    # Test mean calculations
    assert results.mean_creation_time == 2.0
    assert results.mean_compute_time == 5.0

    # Test standard deviation calculations
    assert results.std_creation_time > 0
    assert results.std_compute_time > 0

    # Test empty list edge case
    empty_results = benchmark.Results()
    assert len(empty_results.creation_times) == 0
    assert len(empty_results.compute_times) == 0
    assert empty_results.creation_memory_peak == 0
    assert empty_results.compute_memory_peak == 0
