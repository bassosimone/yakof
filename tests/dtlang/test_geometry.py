"""Tests for the yakof.dtlang.geometry module."""

# SPDX-License-Identifier: Apache-2.0

from yakof.dtlang import geometry
from yakof.frontend import graph


def test_tensor_space():
    """Test the tensor space initialization and basic operations."""
    # Check that space is initialized correctly
    assert geometry.space is not None

    # Test basic tensor operations
    a = geometry.space.placeholder("a")
    b = geometry.space.placeholder("b")

    # Addition
    c = a + b
    assert isinstance(c.node, graph.Node)

    # Multiplication
    d = a * b
    assert isinstance(d.node, graph.Node)

    # Constants
    e = geometry.space.constant(5)
    assert isinstance(e.node, graph.Node)

    # Comparison
    f = geometry.space.less(a, b)
    assert isinstance(f.node, graph.Node)
