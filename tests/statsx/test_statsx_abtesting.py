"""Compatibility tests between yakof.statsx and dt_model."""

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, List, Tuple, Dict, Sequence

import dt_model
import pytest
import numpy as np

from yakof import statsx


@dataclass
class LegacySample:
    """Adapter to match new Sample interface with legacy tuple."""

    weight: float
    value: Any

    @classmethod
    def from_tuple(cls, tup: Tuple[float, Any]) -> "LegacySample":
        """Convert a (probability, value) tuple to a Sample."""
        return cls(weight=tup[0], value=tup[1])



@pytest.mark.usefixtures("fixed_random_seed")
def test_uniform_categorical_sampler():
    """Test that UniformCategoricalSampler matches UniformCategoricalContextVariable."""
    # Setup data
    values = [1, 2, 3, 4, 5]
    support_size = len(values)

    # Create both samplers
    legacy = dt_model.UniformCategoricalContextVariable("test", values)
    new = statsx.UniformCategoricalSampler(values)

    # Test with different sample counts
    for count in [1, 3, 5, 10]:
        # The default behavior should be force_sample=False
        force_sample = False

        # Get samples from both implementations
        legacy_samples = [LegacySample.from_tuple(s) for s in legacy.sample(count)]
        new_samples = new.sample(count)

        # Calculate expected sizes based on behavior
        expected_legacy_size = min(count, support_size) if not force_sample else count
        expected_new_size = min(count, support_size) if not force_sample else count

        # Verify sizes match expected behavior
        assert len(legacy_samples) == expected_legacy_size, f"Legacy size with count={count}"
        assert len(new_samples) == expected_new_size, f"New size with count={count}"

        # Check weights sum to 1 for both
        legacy_weights = [s.weight for s in legacy_samples]
        new_weights = [s.weight for s in new_samples]
        assert sum(legacy_weights) == pytest.approx(1.0)
        assert sum(new_weights) == pytest.approx(1.0)

        # For the count=5 and count=10 cases, both should return all values
        if count >= support_size:
            # All values should be included
            assert {s.value for s in legacy_samples} == set(values)
            assert {s.value for s in new_samples} == set(values)

            # In uniform case, all weights should be 1/support_size
            for s in legacy_samples:
                assert s.weight == pytest.approx(1/support_size)
            for s in new_samples:
                assert s.weight == pytest.approx(1/support_size)


@pytest.mark.usefixtures("fixed_random_seed")
def test_categorical_sampler():
    """Test that CategoricalSampler matches CategoricalContextVariable."""
    # Setup data
    distribution = {1: 0.2, 2: 0.3, 3: 0.5}
    support_size = len(distribution)

    # Create both samplers
    legacy = dt_model.CategoricalContextVariable("test", distribution)
    new = statsx.CategoricalSampler(distribution)

    # Test with different sample counts
    for count in [1, 2, 3, 5]:
        # The default behavior should be force_sample=False
        force_sample = False

        # Get samples from both implementations
        legacy_samples = [LegacySample.from_tuple(s) for s in legacy.sample(count)]
        new_samples = new.sample(count)

        # Calculate expected sizes based on behavior
        expected_legacy_size = min(count, support_size) if not force_sample else count
        expected_new_size = min(count, support_size) if not force_sample else count

        # Verify sizes match expected behavior
        assert len(legacy_samples) == expected_legacy_size, f"Legacy size with count={count}"
        assert len(new_samples) == expected_new_size, f"New size with count={count}"

        # Check weights sum to 1 for both
        legacy_weights = [s.weight for s in legacy_samples]
        new_weights = [s.weight for s in new_samples]
        assert sum(legacy_weights) == pytest.approx(1.0)
        assert sum(new_weights) == pytest.approx(1.0)

        # For count >= support_size, check that original probabilities are preserved
        if count >= support_size:
            # All values should be included
            assert {s.value for s in legacy_samples} == set(distribution.keys())
            assert {s.value for s in new_samples} == set(distribution.keys())

            # Weights should match original distribution
            for s in legacy_samples:
                assert s.weight == pytest.approx(distribution[s.value])
            for s in new_samples:
                assert s.weight == pytest.approx(distribution[s.value])


@pytest.mark.usefixtures("fixed_random_seed")
def test_uniform_categorical_all_values():
    """Test returning all values from uniform distribution."""
    # Setup data
    values = [1, 2, 3]

    # Create both samplers
    legacy = dt_model.UniformCategoricalContextVariable("test", values)
    new = statsx.UniformCategoricalSampler(values)

    # Sample all values (count >= support size)
    legacy_samples = [LegacySample.from_tuple(s) for s in legacy.sample(3)]
    new_samples = new.sample(3)

    # Verify the samples contain all values with uniform weights
    assert len(legacy_samples) == len(new_samples) == 3

    # Check all values are returned with equal probability
    assert {s.value for s in legacy_samples} == set(values)
    assert {s.value for s in new_samples} == set(values)

    # Verify probabilities
    for s in legacy_samples:
        assert s.weight == pytest.approx(1 / 3)
    for s in new_samples:
        assert s.weight == pytest.approx(1 / 3)


@pytest.mark.usefixtures("fixed_random_seed")
def test_categorical_all_values():
    """Test returning all values from categorical distribution."""
    # Setup data
    distribution = {1: 0.2, 2: 0.3, 3: 0.5}

    # Create both samplers
    legacy = dt_model.CategoricalContextVariable("test", distribution)
    new = statsx.CategoricalSampler(distribution)

    # Sample all values (count >= support size)
    legacy_samples = [LegacySample.from_tuple(s) for s in legacy.sample(3)]
    new_samples = new.sample(3)

    # Verify the samples contain all values with original weights
    assert len(legacy_samples) == len(new_samples) == 3

    # Check all values are returned
    legacy_values = {s.value for s in legacy_samples}
    new_values = {s.value for s in new_samples}
    assert legacy_values == new_values == set(distribution.keys())

    # Verify original probabilities are preserved
    for s in legacy_samples:
        assert s.weight == pytest.approx(distribution[s.value])
    for s in new_samples:
        assert s.weight == pytest.approx(distribution[s.value])


@pytest.mark.usefixtures("fixed_random_seed")
def test_categorical_subset():
    """Test sampling from a subset of categorical values."""
    # Setup data
    distribution = {1: 0.2, 2: 0.3, 3: 0.5}
    subset = [1, 3]  # Exclude value 2

    # Create both samplers
    legacy = dt_model.CategoricalContextVariable("test", distribution)
    new = statsx.CategoricalSampler(distribution)

    # Sample from subset
    legacy_samples = [
        LegacySample.from_tuple(s) for s in legacy.sample(2, subset=subset)
    ]
    new_samples = new.sample(count=2, subset=subset)

    # Verify results
    assert len(legacy_samples) == len(new_samples) == 2

    # Check only subset values are returned
    for sample in legacy_samples + list(new_samples):
        assert sample.value in subset

    # Verify weights are normalized
    assert sum(s.weight for s in legacy_samples) == pytest.approx(1.0)
    assert sum(s.weight for s in new_samples) == pytest.approx(1.0)

    # In the full subset case, the weights should be normalized original probabilities
    # Original probabilities: {1: 0.2, 3: 0.5}, normalized: {1: 0.2/0.7, 3: 0.5/0.7}
    expected_1_weight = 0.2 / 0.7
    expected_3_weight = 0.5 / 0.7

    for samples in [legacy_samples, new_samples]:
        for sample in samples:
            if sample.value == 1:
                assert sample.weight == pytest.approx(expected_1_weight)
            elif sample.value == 3:
                assert sample.weight == pytest.approx(expected_3_weight)


@pytest.mark.usefixtures("fixed_random_seed")
def test_continuous_sampler():
    """Test that ContinuousSampler matches ContinuousContextVariable."""
    # Setup continuous distribution
    normal_dist = statsx.scipy_normal(loc=0, scale=1)

    # Create both samplers
    legacy = dt_model.ContinuousContextVariable("test", normal_dist)  # type: ignore
    new = statsx.ContinuousSampler(normal_dist)

    # Test with different sample counts
    for count in [1, 3, 5]:
        # Get samples from both implementations
        legacy_samples = [LegacySample.from_tuple(s) for s in legacy.sample(count)]
        new_samples = new.sample(count)

        # Verify results
        assert len(legacy_samples) == len(new_samples) == count

        # Check weights sum to 1
        assert sum(s.weight for s in legacy_samples) == pytest.approx(1.0)
        assert sum(s.weight for s in new_samples) == pytest.approx(1.0)

        # All samples should have equal weight
        expected_weight = 1.0 / count
        for sample in legacy_samples + list(new_samples):
            assert sample.weight == pytest.approx(expected_weight)


@pytest.mark.usefixtures("fixed_random_seed")
def test_continuous_subset():
    """Test sampling from a subset of continuous values."""
    # Setup continuous distribution
    normal_dist = statsx.scipy_normal(loc=0, scale=1)
    subset = [-1.0, 0.0, 1.0]  # Fixed subset of values

    # Create both samplers
    legacy = dt_model.ContinuousContextVariable("test", normal_dist)  # type: ignore
    new = statsx.ContinuousSampler(normal_dist)

    # Sample from subset
    legacy_samples = [
        LegacySample.from_tuple(s) for s in legacy.sample(3, subset=subset)
    ]
    new_samples = new.sample(count=3, subset=subset)

    # Verify results
    assert len(legacy_samples) == len(new_samples) == 3

    # Check only subset values are returned
    for sample in legacy_samples + list(new_samples):
        assert sample.value in subset

    # Verify weights sum to 1
    assert sum(s.weight for s in legacy_samples) == pytest.approx(1.0)
    assert sum(s.weight for s in new_samples) == pytest.approx(1.0)

    # For normal distribution, the weights should be proportional to PDF values
    pdf_values = [normal_dist.pdf(x) for x in subset]
    total_pdf = sum(pdf_values)
    expected_weights = [pdf / total_pdf for pdf in pdf_values]

    # Check each value has appropriate weight
    for samples in [legacy_samples, new_samples]:
        for sample in samples:
            idx = subset.index(sample.value)
            assert sample.weight == pytest.approx(expected_weights[idx])


@pytest.mark.usefixtures("fixed_random_seed")
def test_force_sample():
    """Test force_sample parameter behavior."""
    # Setup data
    values = [1, 2, 3]

    # Create both samplers
    legacy = dt_model.UniformCategoricalContextVariable("test", values)
    new = statsx.UniformCategoricalSampler(values)

    # Sample with force_sample=True
    legacy_samples = [
        LegacySample.from_tuple(s) for s in legacy.sample(3, force_sample=True)
    ]
    new_samples = new.sample(count=3, force_sample=True)

    # Verify results (should be 3 samples with equal weights)
    assert len(legacy_samples) == len(new_samples) == 3

    # All samples should have equal weight
    for sample in legacy_samples + list(new_samples):
        assert sample.weight == pytest.approx(1 / 3)
