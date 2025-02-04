"""Performance benchmark comparing Yakof against NumPy/SymPy implementations.

This benchmark aims to validate that Yakof's computation graph approach achieves
comparable performance to direct NumPy operations (via SymPy's lambdify). Since
Yakof's main value proposition is improved safety and composability over raw NumPy,
we need high confidence that this doesn't come at a significant performance cost.

The benchmark compares creation time, computation time, and memory usage for both
approaches across different input sizes. Results consistently show that Yakof
achieves reasonably similar performance compared to SymPy-generated NumPy code.
"""

from typing import Callable

import numpy as np
import sympy

from yakof.backend import graph, numpy_engine
from yakof.benchmark import NumericFunc, NumericFuncFactory, Results, run


def build_yakof_model() -> NumericFuncFactory:
    """Build Yakof model factory compatible with benchmark interface.

    Returns a factory function that creates a new instance of the benchmark
    model using Yakof's computation graph. The model computes a ~complex
    mathematical expression combining polynomial, exponential, logarithmic
    and maximum operations.
    """

    def factory() -> NumericFunc:
        # Build the graph
        x = graph.placeholder("x0")
        y = graph.placeholder("x1")
        t1 = x * x + y * y
        t2 = x * y
        t3 = graph.exp(-0.1 * (t1 - t2))
        t4 = graph.log(1.0 + t1 + t2)
        t5 = graph.maximum(t3, t4)
        result = t1 / (1.0 + t2 * t2) + t5 * 0.1

        # Create the evaluation function
        def evaluate(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
            # Note: explicitly using a NullCache to avoid caching results
            ctx = numpy_engine.PartialEvaluationContext(
                bindings={"x0": xx, "x1": yy}, cache=numpy_engine.NullCache()
            )
            return ctx.evaluate(result)

        return evaluate

    return factory


def build_sympy_model() -> NumericFuncFactory:
    """Build SymPy model factory compatible with benchmark interface.

    Returns a factory function that creates a new instance of the benchmark
    model using SymPy's symbolic computation and lambdify. The model computes
    the same mathematical expression used for the Yakof version.
    """

    def factory() -> NumericFunc:
        # Note: we need to sprinkle some `type: ignore` to appease pyright

        # Build the overall expression
        x = sympy.Symbol("x0")
        y = sympy.Symbol("x1")
        t1 = x * x + y * y  # type: ignore
        t2 = x * y  # type: ignore
        t3 = sympy.exp(-0.1 * (t1 - t2))
        t4 = sympy.log(1.0 + t1 + t2)
        t5 = sympy.Max(t3, t4)
        expr = t1 / (1.0 + t2 * t2) + t5 * 0.1  # type: ignore

        # Create the evaluation function
        func = sympy.lambdify([x, y], expr)
        return lambda xx, yy: func(xx, yy)

    return factory


def print_comparison(yakof_results: Results, sympy_results: Results) -> None:
    """Print comparative benchmark results between Yakof and SymPy implementations."""
    print("\nPure computation times:")
    print(
        f"Yakof: {yakof_results.mean_compute_time:.6f} ± "
        f"{yakof_results.std_compute_time:.6f} seconds"
    )
    print(
        f"SymPy: {sympy_results.mean_compute_time:.6f} ± "
        f"{sympy_results.std_compute_time:.6f} seconds"
    )

    print("\nModel creation times:")
    print(
        f"Yakof: {yakof_results.mean_creation_time:.6f} ± "
        f"{yakof_results.std_creation_time:.6f} seconds"
    )
    print(
        f"SymPy: {sympy_results.mean_creation_time:.6f} ± "
        f"{sympy_results.std_creation_time:.6f} seconds"
    )

    print("\nMemory usage:")
    print(
        f"Yakof creation: {yakof_results.creation_memory_peak / (1024 * 1024):.2f} MiB"
    )
    print(f"Yakof compute: {yakof_results.compute_memory_peak / (1024 * 1024):.2f} MiB")
    print(
        f"SymPy creation: {sympy_results.creation_memory_peak / (1024 * 1024):.2f} MiB"
    )
    print(f"SymPy compute: {sympy_results.compute_memory_peak / (1024 * 1024):.2f} MiB")


def main() -> None:
    """Run benchmarks comparing Yakof and SymPy implementations."""
    grid_sizes: list[int] = [100, 500, 1000, 2000]
    n_runs: int = 100

    for size in grid_sizes:
        print(f"\nGrid size: {size}x{size}")
        x = np.linspace(-2, 2, size)
        y = np.linspace(-2, 2, size)

        yakof_results = run(x, y, build_yakof_model(), n_runs=n_runs)
        sympy_results = run(x, y, build_sympy_model(), n_runs=n_runs)

        print_comparison(yakof_results, sympy_results)

        # Verify results match
        #
        # Note: we expect results close to the machine epsilon
        xx, yy = np.meshgrid(x, y)
        yakof_result = build_yakof_model()()(xx, yy)
        sympy_result = build_sympy_model()()(xx, yy)
        max_diff = float(np.max(np.abs(yakof_result - sympy_result)))
        print(f"\nMaximum difference between implementations: {max_diff}")


if __name__ == "__main__":
    main()
