"""Tests for the yakof.dtlang.context module."""

# SPDX-License-Identifier: Apache-2.0

from yakof.dtlang import context, geometry
from yakof.frontend import graph

import pytest


def categorical_context_variable():
    """Make sure CategoricalContextVariable is working as intended."""

    # Make sure the construct raises if the values are empty
    with pytest.raises(ValueError):
        _ = context.CategoricalContextVariable("ccv", {})

    # Create a proper variable
    ccv = context.CategoricalContextVariable("ccv", {"a": 1/3, "b": 1/3, "c": 1/3})

    # Make sure the support size is correct
    assert ccv.support_size() == 3

    # Make sure equality comparison returns a geometry.Tensor
    comparison = ccv == "a"
    assert isinstance(comparison.node, graph.Node)

    # Make sure hashing works
    d = {ccv: 1}
    assert d[ccv] == 1

    # Ensure adaptive sampling is working as intended
    points = ccv.sample(1000)
    assert len(points) == 3

    # Ensure subset sampling is working as intended
    points = ccv.sample(1000, subset=["a", "b"])
    assert len(points) == 2

    # Ensure forced sampling is working as intended
    points = ccv.sample(2, force_sample=True)
    assert len(points) == 1000
