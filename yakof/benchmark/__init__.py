"""
Benchmarking utilities.
======================

This package provides benchmarking tools for numerical computations that operate on
numpy arrays, with a focus on measuring creation and computation performance.

For each benchmark, it measures:
- Creation time: Time taken to build the numeric computation model
- Computation time: Time taken to execute the computation
- Memory usage: Peak memory used during both creation and computation

The main use case is comparing the performance of different numerical computation
backends (e.g. yakof vs sympy) across varying input sizes. Results include both
mean times and standard deviations to assess the performance stability.

See the examples directory for sample benchmark scripts demonstrating usage.
"""

import gc
import timeit
import tracemalloc
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Results:
    """Contains the benchmark results."""

    creation_times: list[float] = field(default_factory=list)
    creation_memory_peak: int = 0

    compute_times: list[float] = field(default_factory=list)
    compute_memory_peak: int = 0

    @property
    def mean_creation_time(self) -> float:
        return float(np.mean(self.creation_times))

    @property
    def mean_compute_time(self) -> float:
        return float(np.mean(self.compute_times))

    @property
    def std_creation_time(self) -> float:
        return float(np.std(self.creation_times))

    @property
    def std_compute_time(self) -> float:
        return float(np.std(self.compute_times))


NumericFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]
NumericFuncFactory = Callable[[], NumericFunc]


def run(x: np.ndarray, y: np.ndarray, factory: NumericFuncFactory, n_runs: int = 100) -> Results:
    """Benchmarks the given factory function using the given x and y arrays.

    The factory function should return a callable that implements a numerical computation
    on two numpy arrays. The returned callable should take two arguments (x and y arrays)
    and return a numpy array containing the computation results.

    Example factory function:
        def my_factory():
            def compute(x: np.ndarray, y: np.ndarray) -> np.ndarray:
                return x + y  # Some numerical computation
            return compute

    Args:
        x: First input array for the computation
        y: Second input array for the computation
        factory: A function that returns a callable implementing the numerical computation
        n_runs: Number of times to repeat the benchmark measurements

    Returns
    -------
        A Results object containing timing and memory measurements
    """
    # Create the final results
    results = Results()

    # Create test data using a meshgrid for proper broadcasting
    xx, yy = np.meshgrid(x, y)

    # Measure the overall creation memory
    tracemalloc.start()
    numeric_func = factory()
    _, peak_memory = tracemalloc.get_traced_memory()  # only focus on peak
    tracemalloc.stop()
    results.creation_memory_peak = peak_memory

    # Measure the computation memory
    tracemalloc.start()
    _ = numeric_func(xx, yy)
    _, peak_memory = tracemalloc.get_traced_memory()  # only focus on peak
    tracemalloc.stop()
    results.compute_memory_peak = peak_memory

    # Benchmark runs
    for _ in range(n_runs):
        # Measure the creation time
        gc.collect()
        create_start = timeit.default_timer()
        numeric_func = factory()
        create_time = timeit.default_timer() - create_start
        results.creation_times.append(create_time)

        # Measure the computation time
        gc.collect()
        compute_start = timeit.default_timer()
        _ = numeric_func(xx, yy)
        compute_time = timeit.default_timer() - compute_start
        results.compute_times.append(compute_time)

    return results
