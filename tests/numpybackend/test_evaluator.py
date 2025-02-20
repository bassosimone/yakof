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


def test_conditional_operations():
    """Test evaluation of conditional operations."""
    # Test simple where operation first
    cond = np.array([True, False, True])
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    bindings = {"cond": cond, "x": x, "y": y}

    # Simple where
    node = graph.where(
        graph.placeholder("cond"), graph.placeholder("x"), graph.placeholder("y")
    )
    result = evaluator.evaluate(node, bindings)
    expected = np.where(cond, x, y)
    assert np.array_equal(result, expected)

    # Print intermediate values for debugging
    print(f"cond: {cond}")
    print(f"x: {x}")
    print(f"y: {y}")
    print(f"result: {result}")
    print(f"expected: {expected}")


def test_caching():
    """Test that evaluation caching works correctly."""
    # Create a simple expression that we can verify caching with
    x = graph.placeholder("x")
    const = graph.constant(2.0)
    expr = graph.multiply(x, const)  # x * 2

    # First evaluation
    cache = {}
    x_val = np.array(3.0)
    result1 = evaluator.evaluate(expr, {"x": x_val}, cache)

    # Second evaluation - should use cache
    result2 = evaluator.evaluate(expr, {"x": x_val}, cache)

    # Results should be identical
    assert np.array_equal(result1, result2)
    assert np.array_equal(result1, x_val * 2)

    # Print cache contents for debugging
    print("Cache contents:")
    for k, v in cache.items():
        print(f"  {k}: {v}")


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
