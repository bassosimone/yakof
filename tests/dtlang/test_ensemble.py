"""Tests for the yakof.dtlang.ensemble module."""

# SPDX-License-Identifier: Apache-2.0


from yakof.dtlang import context


def test_ensemble_types():
    """Test the ensemble type definitions."""
    from yakof.dtlang import ensemble

    # Test Weight type
    weight: ensemble.Weight = 0.5
    assert isinstance(weight, float)

    # Test Variables type with a simple example
    cv = context.UniformCategoricalContextVariable("test", ["a", "b"])
    variables: ensemble.Variables = {cv: 1.0}
    assert len(variables) == 1
    assert cv in variables
    assert variables[cv] == 1.0
