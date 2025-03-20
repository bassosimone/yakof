"""Tests for the yakof.dtlang.context module."""

# SPDX-License-Identifier: Apache-2.0

import pytest

from yakof.dtlang import context
from yakof.frontend import graph


def test_categorical_context_variable():
    """Make sure CategoricalContextVariable is working as intended."""

    # Make sure the construct raises if the values are empty
    with pytest.raises(ValueError):
        _ = context.CategoricalContextVariable("ccv", {})

    # Create a proper variable
    ccv = context.CategoricalContextVariable("ccv", {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})

    # Make sure the support size is correct
    assert ccv.support_size() == 3

    # Make sure equality comparison returns a geometry.Tensor
    comparison = ccv == "a"
    assert isinstance(comparison.node, graph.Node)

    # Make sure hashing works
    d = {ccv: 1}
    assert d[ccv] == 1

    # Ensure adaptive sampling is working as intended
    samples = ccv.sample(1000)
    assert len(samples) == 3
    assert isinstance(samples[0][0], float)  # weight
    assert isinstance(samples[0][1], float)  # value

    # Ensure subset sampling is working as intended
    subset_samples = ccv.sample(1000, subset=["a", "b"])
    assert len(subset_samples) == 2

    # Ensure forced sampling is working as intended
    forced_samples = ccv.sample(1000, force_sample=True)
    assert len(forced_samples) == 1000


def test_uniform_categorical_context_variable():
    """Test UniformCategoricalContextVariable initialization and behavior."""
    # Make sure the construct raises if the values are empty
    with pytest.raises(ValueError):
        _ = context.UniformCategoricalContextVariable("ucv", [])

    # Create a proper variable
    ucv = context.UniformCategoricalContextVariable("ucv", ["a", "b", "c"])

    # Make sure the support size is correct
    assert ucv.support_size() == 3

    # Make sure equality comparison returns a geometry.Tensor
    comparison = ucv == "a"
    assert isinstance(comparison.node, graph.Node)

    # Make sure hashing works
    d = {ucv: 1}
    assert d[ucv] == 1

    # Make sure adaptive sampling is working as intended
    samples = ucv.sample(3)
    assert len(samples) == 3
    assert isinstance(samples[0][0], float)  # weight
    assert isinstance(samples[0][1], float)  # value

    # Check sampling with subset
    subset_samples = ucv.sample(1, subset=["a", "b"])
    assert len(subset_samples) == 1

    # Test forced sampling
    forced_samples = ucv.sample(2, force_sample=True)
    assert len(forced_samples) == 2
