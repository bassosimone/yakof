"""Tests for the yakof.dtlang.piecewise module."""

# SPDX-License-Identifier: Apache-2.0

from yakof.dtlang import geometry, piecewise
from yakof.frontend import linearize
from yakof.numpybackend import executor

import numpy as np


def test_piecewise():
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
    state = executor.State({
        p_expr1.node: expr1,
        p_expr2.node: expr2,
        p_expr3.node: expr3,
        p_filter1.node: filter1,
        p_filter2.node: filter2,
    })

    # Actually evaluate the piecewise function
    for node in plan:
        executor.evaluate(state, node)

    # Ensure the result is the expected one
    expect = np.array([2, 27, 256])
    rv = state.values[pw.node]
    assert len(rv) == len(expect)
    for idx in range(len(expect)):
        assert rv[idx] == expect[idx]
