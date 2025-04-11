"""Tests for the yakof.dtlang.piecewise module."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from yakof.dtlang import geometry, piecewise
from yakof.frontend import linearize
from yakof.numpybackend import executor


def test_piecewise_basics():
    """Make sure that Piecewise works as intended."""
    # Create the expressions
    expr1 = np.array([2, 9, 16])
    expr2 = np.array([8, 27, 64])
    expr3 = np.array([16, 81, 256])

    # Create the filters
    filter1 = np.array([True, False, False])
    filter2 = np.array([False, True, False])

    # Create the placeholders
    p_expr1 = geometry.space.placeholder("expr1")
    p_expr2 = geometry.space.placeholder("expr2")
    p_expr3 = geometry.space.placeholder("expr3")
    p_filter1 = geometry.space.placeholder("filter1")
    p_filter2 = geometry.space.placeholder("filter2")

    # Create the resulting piecewise function
    pw = piecewise.Piecewise(
        (p_expr1, p_filter1),
        (p_expr2, p_filter2),
        (p_expr3, True),
    )

    # Linearize an execution plan out of the piecewise
    plan = linearize.forest(pw.node)
    assert len(plan) == 6

    # Create the evaluation state
    state = executor.State(
        {
            p_expr1.node: expr1,
            p_expr2.node: expr2,
            p_expr3.node: expr3,
            p_filter1.node: filter1,
            p_filter2.node: filter2,
        }
    )

    # Actually evaluate the piecewise function
    for node in plan:
        executor.evaluate(state, node)

    # Ensure the result is the expected one
    expect = np.array([2, 27, 256])
    rv = state.values[pw.node]
    assert np.all(rv == expect)


def test_piecewise_with_scalars():
    """Test Piecewise with scalar values."""
    # Simple case with scalar values
    result = piecewise.Piecewise((1, geometry.space.constant(True)), (2, geometry.space.constant(False)))

    # Linearize
    plan = linearize.forest(result.node)
    state = executor.State(values={})

    # Evaluate
    for node in plan:
        executor.evaluate(state, node)

    assert state.values[result.node] == 1


def test_piecewise_empty():
    """Test Piecewise with no clauses raises ValueError."""
    with pytest.raises(ValueError):
        piecewise.Piecewise()


def test_piecewise_filtering():
    """Test the internal _filter_clauses function."""
    clauses = (
        (1, False),
        (2, True),
        (3, False),  # Should be filtered out
        (4, True),  # Should be filtered out
    )

    filtered = piecewise._filter_clauses(clauses)
    assert len(filtered) == 2
    assert filtered[0] == (1, False)
    assert filtered[1] == (2, True)


def test_piecewise_with_constant_conditions():
    """Test Piecewise functionality with constant conditions."""
    # Create expression tensors
    expr1 = geometry.space.placeholder("expr1")
    expr2 = geometry.space.placeholder("expr2")
    expr3 = geometry.space.placeholder("expr3")

    # Create piecewise with constant conditions
    pw = piecewise.Piecewise(
        (expr1, False),  # Constant False condition
        (expr2, True),  # Constant True condition
        (expr3, True),  # This should be ignored as it's after a True condition
    )

    # Linearize the execution plan
    plan = linearize.forest(pw.node)

    # Set up evaluation state with tensor values
    state = executor.State(
        {
            expr1.node: np.array([10, 20, 30]),
            expr2.node: np.array([40, 50, 60]),
            expr3.node: np.array([70, 80, 90]),
        }
    )

    # Evaluate the piecewise function
    for node in plan:
        executor.evaluate(state, node)

    # Since the second condition is True, the result should be expr2
    result = state.values[pw.node]
    expected = np.array([40, 50, 60])

    assert np.array_equal(result, expected)

    # Test with just default case (single True condition) and a false condition
    # to avoid empty condition list error
    pw_default = piecewise.Piecewise(
        (expr2, False),  # False condition (needed to avoid empty condition list)
        (expr1, True),  # Default case
    )

    plan_default = linearize.forest(pw_default.node)
    state_default = executor.State(
        {
            expr1.node: np.array([10, 20, 30]),
            expr2.node: np.array([40, 50, 60]),
        }
    )

    for node in plan_default:
        executor.evaluate(state_default, node)

    result_default = state_default.values[pw_default.node]
    assert np.array_equal(result_default, np.array([10, 20, 30]))
