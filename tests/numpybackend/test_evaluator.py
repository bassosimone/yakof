"""Tests for the yakof.numpybackend.evaluator module."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from yakof.frontend import graph
from yakof.numpybackend import evaluator


def test_constant_evaluation():
    """Test evaluation of constant nodes."""
    # Scalar constants
    assert np.array_equal(evaluator.evaluate(graph.constant(1.0), {}), np.array(1.0))
    assert np.array_equal(evaluator.evaluate(graph.constant(True), {}), np.array(True))
    assert np.array_equal(evaluator.evaluate(graph.constant(42), {}), np.array(42))


def test_placeholder_evaluation():
    """Test evaluation of placeholder nodes."""
    x = graph.placeholder("x")
    y = graph.placeholder("y", default_value=3.14)

    # Test with binding
    assert np.array_equal(evaluator.evaluate(x, {"x": np.array(2.0)}), np.array(2.0))

    # Test default value
    assert np.array_equal(evaluator.evaluate(y, {}), np.array(3.14))

    # Test missing binding
    with pytest.raises(ValueError, match="no value provided for placeholder 'x'"):
        evaluator.evaluate(x, {})


def test_arithmetic_operations():
    """Test evaluation of arithmetic operations."""
    x = np.array(2.0)
    y = np.array(3.0)
    bindings = {"x": x, "y": y}

    # Addition
    node = graph.add(graph.placeholder("x"), graph.placeholder("y"))
    assert np.array_equal(evaluator.evaluate(node, bindings), x + y)

    # Subtraction
    node = graph.subtract(graph.placeholder("x"), graph.placeholder("y"))
    assert np.array_equal(evaluator.evaluate(node, bindings), x - y)

    # Multiplication
    node = graph.multiply(graph.placeholder("x"), graph.placeholder("y"))
    assert np.array_equal(evaluator.evaluate(node, bindings), x * y)

    # Division
    node = graph.divide(graph.placeholder("x"), graph.placeholder("y"))
    assert np.array_equal(evaluator.evaluate(node, bindings), x / y)


def test_comparison_operations():
    """Test evaluation of comparison operations."""
    x = np.array(2.0)
    y = np.array(3.0)
    bindings = {"x": x, "y": y}

    # Less than
    node = graph.less(graph.placeholder("x"), graph.placeholder("y"))
    assert np.array_equal(evaluator.evaluate(node, bindings), x < y)

    # Greater than
    node = graph.greater(graph.placeholder("x"), graph.placeholder("y"))
    assert np.array_equal(evaluator.evaluate(node, bindings), x > y)

    # Equal
    node = graph.equal(graph.placeholder("x"), graph.placeholder("y"))
    assert np.array_equal(evaluator.evaluate(node, bindings), x == y)


def test_logical_operations():
    """Test evaluation of logical operations."""
    x = np.array(True)
    y = np.array(False)
    bindings = {"x": x, "y": y}

    # AND
    node = graph.logical_and(graph.placeholder("x"), graph.placeholder("y"))
    assert np.array_equal(evaluator.evaluate(node, bindings), np.logical_and(x, y))

    # OR
    node = graph.logical_or(graph.placeholder("x"), graph.placeholder("y"))
    assert np.array_equal(evaluator.evaluate(node, bindings), np.logical_or(x, y))

    # NOT
    node = graph.logical_not(graph.placeholder("x"))
    assert np.array_equal(evaluator.evaluate(node, bindings), np.logical_not(x))


def test_math_operations():
    """Test evaluation of mathematical operations."""
    x = np.array(2.0)
    bindings = {"x": x}

    # Exponential
    node = graph.exp(graph.placeholder("x"))
    assert np.array_equal(evaluator.evaluate(node, bindings), np.exp(x))

    # Logarithm
    node = graph.log(graph.placeholder("x"))
    assert np.array_equal(evaluator.evaluate(node, bindings), np.log(x))

    # Power
    node = graph.power(graph.placeholder("x"), graph.constant(2.0))
    assert np.array_equal(evaluator.evaluate(node, bindings), np.power(x, 2.0))


def test_broadcasting():
    """Test broadcasting behavior in operations."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([10.0, 20.0])
    bindings = {"x": x, "y": y}

    # Broadcasting in addition
    node = graph.add(graph.placeholder("x"), graph.placeholder("y"))
    assert np.array_equal(evaluator.evaluate(node, bindings), x + y)


def test_reduction_operations():
    """Test evaluation of reduction operations."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    bindings = {"x": x}

    # Sum reduction
    node = graph.reduce_sum(graph.placeholder("x"), axis=0)
    assert np.array_equal(evaluator.evaluate(node, bindings), np.sum(x, axis=0))

    # Mean reduction
    node = graph.reduce_mean(graph.placeholder("x"), axis=1)
    assert np.array_equal(evaluator.evaluate(node, bindings), np.mean(x, axis=1))


def test_where_operation():
    """Test where operation with various data types and shapes."""
    test_cases = [
        # Basic scalar selection
        {
            "cond": np.array([True, False, True]),
            "x": np.array([1.0, 2.0, 3.0]),
            "y": np.array([4.0, 5.0, 6.0]),
            "desc": "scalar values",
        },
        # Boolean values
        {
            "cond": np.array([True, False, True]),
            "x": np.array([True, True, True]),
            "y": np.array([False, False, False]),
            "desc": "boolean values",
        },
        # 2D array broadcasting
        {
            "cond": np.array([[True, False], [False, True]]),
            "x": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "y": np.array([[5.0, 6.0], [7.0, 8.0]]),
            "desc": "2D arrays",
        },
    ]

    for case in test_cases:
        print(f"\nTesting where with {case['desc']}:")
        bindings = {"cond": case["cond"], "x": case["x"], "y": case["y"]}

        node = graph.where(
            graph.placeholder("cond"), graph.placeholder("x"), graph.placeholder("y")
        )
        result = evaluator.evaluate(node, bindings)
        expected = np.where(case["cond"], case["x"], case["y"])

        print(f"condition: {case['cond']}")
        print(f"x: {case['x']}")
        print(f"y: {case['y']}")
        print(f"result: {result}")
        print(f"expected: {expected}")

        assert np.array_equal(result, expected)


def test_debug_operations(capsys):
    """Test debug operations (tracepoint/breakpoint)."""
    x = graph.placeholder("x")
    traced = graph.tracepoint(x)

    evaluator.evaluate(traced, {"x": np.array(1.0)})
    captured = capsys.readouterr()
    assert "=== begin tracepoint ===" in captured.out
    assert "value" in captured.out


def test_error_handling():
    """Test various error conditions."""

    # Unknown operation type
    class unknown_op(graph.Node):
        pass

    with pytest.raises(TypeError, match="unknown node type"):
        evaluator.evaluate(unknown_op(), {})


def test_multi_clause_where():
    """Test multi-clause where operations, comparing with both np.select and nested where."""
    # Setup test conditions
    cond1 = np.array([True, False, False])
    cond2 = np.array([False, True, False])
    val1, val2 = 1.0, 2.0
    default = 0.0
    bindings = {"cond1": cond1, "cond2": cond2}

    # Test our multi_clause_where implementation
    node = graph.multi_clause_where(
        [
            (graph.placeholder("cond1"), graph.constant(val1)),
            (graph.placeholder("cond2"), graph.constant(val2)),
        ],
        graph.constant(default),
    )
    result = evaluator.evaluate(node, bindings)

    # Compare with np.select
    expected_select = np.select([cond1, cond2], [val1, val2], default=default)

    # Compare with nested where operations
    nested_where = graph.where(
        graph.placeholder("cond1"),
        graph.constant(val1),
        graph.where(
            graph.placeholder("cond2"), graph.constant(val2), graph.constant(default)
        ),
    )
    expected_nested = evaluator.evaluate(nested_where, bindings)

    # Debug information
    print("\nDebug information:")
    print(f"cond1: {cond1}")
    print(f"cond2: {cond2}")
    print(f"result from multi_clause_where: {result}")
    print(f"expected from np.select: {expected_select}")
    print(f"expected from nested where: {expected_nested}")

    # Verify all approaches give same result
    assert np.array_equal(result, expected_select)
    assert np.array_equal(result, expected_nested)


def test_caching_behavior():
    """Test caching behavior."""
    # Create a computation we can verify
    x = graph.placeholder("x")
    node = graph.add(x, x)  # x + x

    # Set up cache and bindings
    cache = {}
    value = np.array(2.0)
    bindings = {"x": value}

    # First evaluation should compute and cache
    result1 = evaluator.evaluate(node, bindings, cache=cache)

    # Modify bindings to ensure we're using cache
    bindings["x"] = np.array(3.0)  # Change input value

    # Second evaluation with same node should use cached value
    result2 = evaluator.evaluate(node, bindings, cache=cache)

    # Results should be identical despite different input
    assert np.array_equal(result1, result2)
    assert np.array_equal(result1, value + value)  # Should be 2.0 + 2.0
