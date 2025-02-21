"""Tests for the yakof.numpybackend.evaluator module."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from yakof.frontend import graph
from yakof.numpybackend import evaluator


def test_constant_evaluation():
    """Test evaluation of constant nodes."""
    # Scalar constants
    state = evaluator.StateWithoutCache({})
    assert np.array_equal(evaluator.evaluate(graph.constant(1.0), state), np.array(1.0))
    assert np.array_equal(
        evaluator.evaluate(graph.constant(True), state), np.array(True)
    )
    assert np.array_equal(evaluator.evaluate(graph.constant(42), state), np.array(42))


def test_placeholder_evaluation():
    """Test evaluation of placeholder nodes."""
    x = graph.placeholder("x")
    y = graph.placeholder("y", default_value=3.14)

    # Test with binding
    state = evaluator.StateWithoutCache(
        {"x": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    )
    assert np.array_equal(
        evaluator.evaluate(x, state), np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    )

    # Test default value
    state = evaluator.StateWithoutCache({})
    assert np.array_equal(evaluator.evaluate(y, state), np.array(3.14))

    # Test missing binding
    with pytest.raises(ValueError, match="no value provided for placeholder 'x'"):
        evaluator.evaluate(x, evaluator.StateWithoutCache({}))


def test_arithmetic_operations():
    """Test evaluation of arithmetic operations."""
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])  # 3x3
    y = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])  # 3x3
    state = evaluator.StateWithoutCache({"x": x, "y": y})

    # Addition
    node = graph.add(graph.placeholder("x"), graph.placeholder("y"))
    assert np.array_equal(evaluator.evaluate(node, state), x + y)

    # Subtraction
    node = graph.subtract(graph.placeholder("x"), graph.placeholder("y"))
    assert np.array_equal(evaluator.evaluate(node, state), x - y)

    # Multiplication
    node = graph.multiply(graph.placeholder("x"), graph.placeholder("y"))
    assert np.array_equal(evaluator.evaluate(node, state), x * y)

    # Division
    node = graph.divide(graph.placeholder("x"), graph.placeholder("y"))
    assert np.array_equal(evaluator.evaluate(node, state), x / y)


def test_comparison_operations():
    """Test evaluation of comparison operations."""
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2x3
    y = np.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]])  # 2x3
    state = evaluator.StateWithoutCache({"x": x, "y": y})

    # Less than
    node = graph.less(graph.placeholder("x"), graph.placeholder("y"))
    assert np.array_equal(evaluator.evaluate(node, state), x < y)

    # Greater than
    node = graph.greater(graph.placeholder("x"), graph.placeholder("y"))
    assert np.array_equal(evaluator.evaluate(node, state), x > y)

    # Equal
    node = graph.equal(graph.placeholder("x"), graph.placeholder("y"))
    assert np.array_equal(evaluator.evaluate(node, state), x == y)


def test_logical_operations():
    """Test evaluation of logical operations."""
    x = np.array([[True, False, True], [False, True, False]])  # 2x3
    y = np.array([[False, True, False], [True, False, True]])  # 2x3
    state = evaluator.StateWithoutCache({"x": x, "y": y})

    # AND
    node = graph.logical_and(graph.placeholder("x"), graph.placeholder("y"))
    assert np.array_equal(evaluator.evaluate(node, state), np.logical_and(x, y))

    # OR
    node = graph.logical_or(graph.placeholder("x"), graph.placeholder("y"))
    assert np.array_equal(evaluator.evaluate(node, state), np.logical_or(x, y))

    # NOT
    node = graph.logical_not(graph.placeholder("x"))
    assert np.array_equal(evaluator.evaluate(node, state), np.logical_not(x))


def test_math_operations():
    """Test evaluation of mathematical operations."""
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2x3
    state = evaluator.StateWithoutCache({"x": x})

    # Exponential
    node = graph.exp(graph.placeholder("x"))
    assert np.array_equal(evaluator.evaluate(node, state), np.exp(x))

    # Logarithm
    node = graph.log(graph.placeholder("x"))
    assert np.array_equal(evaluator.evaluate(node, state), np.log(x))

    # Power
    node = graph.power(graph.placeholder("x"), graph.constant(2.0))
    assert np.array_equal(evaluator.evaluate(node, state), np.power(x, 2.0))


def test_broadcasting():
    """Test broadcasting behavior in operations."""
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])  # 3x3
    y = np.array([10.0, 20.0, 30.0])  # 1x3 for broadcasting
    z = np.array([[1.0], [2.0], [3.0]])  # 3x1 for broadcasting
    state = evaluator.StateWithoutCache({"x": x, "y": y, "z": z})

    # Broadcasting row vector
    node = graph.add(graph.placeholder("x"), graph.placeholder("y"))
    assert np.array_equal(evaluator.evaluate(node, state), x + y)

    # Broadcasting column vector
    node = graph.multiply(graph.placeholder("x"), graph.placeholder("z"))
    assert np.array_equal(evaluator.evaluate(node, state), x * z)


def test_reduction_operations():
    """Test evaluation of reduction operations."""
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])  # 3x3
    state = evaluator.StateWithoutCache({"x": x})

    # Sum reduction over rows
    node = graph.reduce_sum(graph.placeholder("x"), axis=0)
    assert np.array_equal(evaluator.evaluate(node, state), np.sum(x, axis=0))

    # Mean reduction over columns
    node = graph.reduce_mean(graph.placeholder("x"), axis=1)
    assert np.array_equal(evaluator.evaluate(node, state), np.mean(x, axis=1))


def test_where_operation():
    """Test where operation with various data types and shapes."""
    test_cases = [
        # 3x3 matrix selection
        {
            "cond": np.array(
                [[True, False, True], [False, True, False], [True, False, True]]
            ),
            "x": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            "y": np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]),
            "desc": "3x3 matrices",
        },
        # Boolean matrix operations
        {
            "cond": np.array([[True, False], [False, True], [True, False]]),
            "x": np.array([[True, True], [True, True], [True, True]]),
            "y": np.array([[False, False], [False, False], [False, False]]),
            "desc": "3x2 boolean matrices",
        },
        # Broadcasting with column vector
        {
            "cond": np.array([[True], [False], [True]]),
            "x": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            "y": np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
            "desc": "3x2 matrices with broadcast",
        },
    ]

    for case in test_cases:
        print(f"\nTesting where with {case['desc']}:")
        state = evaluator.StateWithoutCache(
            {"cond": case["cond"], "x": case["x"], "y": case["y"]}
        )

        node = graph.where(
            graph.placeholder("cond"), graph.placeholder("x"), graph.placeholder("y")
        )
        result = evaluator.evaluate(node, state)
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

    state = evaluator.StateWithoutCache({"x": np.array([[1.0, 2.0], [3.0, 4.0]])})
    evaluator.evaluate(traced, state)
    captured = capsys.readouterr()
    assert "=== begin tracepoint ===" in captured.out
    assert "value" in captured.out


def test_error_handling():
    """Test various error conditions."""

    # Unknown operation type
    class unknown_op(graph.Node):
        pass

    with pytest.raises(TypeError, match="unknown node type"):
        evaluator.evaluate(unknown_op(), evaluator.StateWithoutCache({}))


def test_multi_clause_where():
    """Test multi-clause where operations, comparing with both np.select and nested where."""
    # Setup test conditions
    cond1 = np.array([[True, False, False], [False, True, False], [False, False, True]])
    cond2 = np.array([[False, True, False], [True, False, False], [False, True, False]])
    val1, val2 = 1.0, 2.0
    default = 0.0
    state = evaluator.StateWithoutCache({"cond1": cond1, "cond2": cond2})

    # Test our multi_clause_where implementation
    node = graph.multi_clause_where(
        [
            (graph.placeholder("cond1"), graph.constant(val1)),
            (graph.placeholder("cond2"), graph.constant(val2)),
        ],
        graph.constant(default),
    )
    result = evaluator.evaluate(node, state)

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
    expected_nested = evaluator.evaluate(nested_where, state)

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
    exp_x = graph.exp(x)  # Use exp(x) so we can verify cache usage
    node = graph.add(exp_x, exp_x)  # exp(x) + exp(x)

    # Set up state with initial value
    initial_value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    state = evaluator.StateWithCache({"x": initial_value})

    # First evaluation should compute and cache
    result1 = evaluator.evaluate(node, state)

    # Modify binding - this should invalidate cache
    new_value = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    state.set_placeholder_value("x", new_value)

    # Second evaluation should compute new result
    result2 = evaluator.evaluate(node, state)

    # Results should be different because input changed
    assert not np.array_equal(result1, result2)

    # Verify results are correct
    assert np.array_equal(result1, np.exp(initial_value) + np.exp(initial_value))
    assert np.array_equal(result2, np.exp(new_value) + np.exp(new_value))

    # Verify that repeated evaluation with same input gives same result
    result3 = evaluator.evaluate(node, state)
    assert np.array_equal(result2, result3)


def test_expand_dims_operation():
    """Test evaluation of expand_dims operation."""
    # Test expanding various dimensions
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2x3
    state = evaluator.StateWithoutCache({"x": x})

    # Add dimension at the beginning (make 1x2x3)
    node = graph.expand_dims(graph.placeholder("x"), axis=0)
    result = evaluator.evaluate(node, state)
    assert result.shape == (1, 2, 3)
    assert np.array_equal(result[0], x)

    # Add dimension in the middle (make 2x1x3)
    node = graph.expand_dims(graph.placeholder("x"), axis=1)
    result = evaluator.evaluate(node, state)
    assert result.shape == (2, 1, 3)
    assert np.array_equal(result[:, 0, :], x)

    # Add dimension at the end (make 2x3x1)
    node = graph.expand_dims(graph.placeholder("x"), axis=2)
    result = evaluator.evaluate(node, state)
    assert result.shape == (2, 3, 1)
    assert np.array_equal(result[:, :, 0], x)


def test_unknown_operations():
    """Test handling of unknown operations."""

    # Unknown binary operation
    class unknown_binary(graph.BinaryOp):
        pass

    with pytest.raises(TypeError, match="unknown binary operation"):
        evaluator.evaluate(
            unknown_binary(graph.constant(1.0), graph.constant(2.0)),
            evaluator.StateWithoutCache({}),
        )

    # Unknown unary operation
    class unknown_unary(graph.UnaryOp):
        pass

    with pytest.raises(TypeError, match="unknown unary operation"):
        evaluator.evaluate(
            unknown_unary(graph.constant(1.0)), evaluator.StateWithoutCache({})
        )

    # Unknown axis operation
    class unknown_axis(graph.AxisOp):
        pass

    with pytest.raises(TypeError, match="unknown axis operation"):
        evaluator.evaluate(
            unknown_axis(graph.constant(1.0), 0), evaluator.StateWithoutCache({})
        )


def test_breakpoint_operation(monkeypatch):
    """Test breakpoint operation using mocked input."""
    # Mock the input function to avoid waiting for user input
    mock_input_calls = []

    def mock_input(prompt):
        mock_input_calls.append(prompt)
        return ""

    monkeypatch.setattr("builtins.input", mock_input)

    # Create and evaluate a breakpointed node
    x = graph.placeholder("x")
    broken = graph.breakpoint(x)

    state = evaluator.StateWithoutCache({"x": np.array([[1.0, 2.0], [3.0, 4.0]])})
    result = evaluator.evaluate(broken, state)

    # Verify breakpoint was triggered
    assert len(mock_input_calls) == 1
    assert mock_input_calls[0] == "Press any key to continue..."
