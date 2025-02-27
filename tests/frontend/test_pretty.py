"""Tests for the yakof.frontend.pretty module."""

# SPDX-License-Identifier: Apache-2.0

import pytest
from yakof.frontend import graph, pretty


def test_basic_pretty_printing():
    """Test basic pretty printing of computation graphs."""
    x = graph.placeholder("x")
    y = graph.constant(2.0)
    z = graph.add(x, y)

    result = pretty.format(z)
    assert result == "x + 2.0"


def test_precedence_pretty_printing():
    """Test pretty printing with operator precedence."""
    x = graph.placeholder("x")
    y = graph.constant(2.0)
    z = graph.constant(3.0)

    expr1 = graph.add(x, graph.multiply(y, z))
    result1 = pretty.format(expr1)
    assert result1 == "x + 2.0 * 3.0"

    expr2 = graph.multiply(graph.add(x, y), z)
    result2 = pretty.format(expr2)
    assert result2 == "(x + 2.0) * 3.0"


def test_unary_operations_pretty_printing():
    """Test pretty printing of unary operations."""
    x = graph.placeholder("x")

    expr1 = graph.exp(x)
    result1 = pretty.format(expr1)
    assert result1 == "exp(x)"

    expr2 = graph.log(x)
    result2 = pretty.format(expr2)
    assert result2 == "log(x)"

    expr3 = graph.logical_not(x)
    result3 = pretty.format(expr3)
    assert result3 == "~x"


def test_comparison_operations_pretty_printing():
    """Test pretty printing of comparison operations."""
    x = graph.placeholder("x")
    y = graph.constant(2.0)

    expr1 = graph.less(x, y)
    result1 = pretty.format(expr1)
    assert result1 == "x < 2.0"

    expr2 = graph.greater_equal(x, y)
    result2 = pretty.format(expr2)
    assert result2 == "x >= 2.0"


def test_logical_operations_pretty_printing():
    """Test pretty printing of logical operations."""
    x = graph.placeholder("x")
    y = graph.constant(True)

    expr1 = graph.logical_and(x, y)
    result1 = pretty.format(expr1)
    assert result1 == "x & True"

    expr2 = graph.logical_or(x, y)
    result2 = pretty.format(expr2)
    assert result2 == "x | True"

    expr3 = graph.logical_xor(x, y)
    result3 = pretty.format(expr3)
    assert result3 == "x ^ True"


def test_named_expressions_pretty_printing():
    """Test pretty printing of named expressions."""
    x = graph.placeholder("x")
    y = graph.constant(2.0, name="const_y")
    z = graph.add(x, y)

    result = pretty.format(z)
    assert result == "x + const_y"


def test_subtract_pretty_printing():
    """Test pretty printing of subtract operation."""
    x = graph.placeholder("x")
    y = graph.constant(2.0)
    z = graph.subtract(x, y)

    result = pretty.format(z)
    assert result == "x - 2.0"


def test_divide_pretty_printing():
    """Test pretty printing of divide operation."""
    x = graph.placeholder("x")
    y = graph.constant(2.0)
    z = graph.divide(x, y)

    result = pretty.format(z)
    assert result == "x / 2.0"


def test_power_pretty_printing():
    """Test pretty printing of power operation."""
    x = graph.placeholder("x")
    y = graph.constant(2.0)
    z = graph.power(x, y)

    result = pretty.format(z)
    assert result == "x ** 2.0"


def test_less_equal_pretty_printing():
    """Test pretty printing of less_equal operation."""
    x = graph.placeholder("x")
    y = graph.constant(2.0)
    z = graph.less_equal(x, y)

    result = pretty.format(z)
    assert result == "x <= 2.0"


def test_greater_pretty_printing():
    """Test pretty printing of greater operation."""
    x = graph.placeholder("x")
    y = graph.constant(2.0)
    z = graph.greater(x, y)

    result = pretty.format(z)
    assert result == "x > 2.0"


def test_equal_pretty_printing():
    """Test pretty printing of equal operation."""
    x = graph.placeholder("x")
    y = graph.constant(2.0)
    z = graph.equal(x, y)

    result = pretty.format(z)
    assert result == "x == 2.0"


def test_not_equal_pretty_printing():
    """Test pretty printing of not_equal operation."""
    x = graph.placeholder("x")
    y = graph.constant(2.0)
    z = graph.not_equal(x, y)

    result = pretty.format(z)
    assert result == "x != 2.0"


def test_unhandled_node_type():
    """Test pretty printing of an unhandled node type."""

    class UnhandledNode(graph.Node):
        pass

    x = UnhandledNode()
    result = pretty.format(x)
    assert result == "<unknown:UnhandledNode>"
