"""Tests for the yakof.minisimulator package."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from yakof import minisimulator
from dt_model.engine.frontend import graph


class TestLinearRange:
    def test_init(self):
        # Test successful initialization
        lr = minisimulator.LinearRange(0, 10, 5)
        assert lr.start == 0
        assert lr.stop == 10
        assert lr.points == 5

    def test_validation(self):
        # Test invalid parameters
        with pytest.raises(ValueError, match="stop must be greater than start"):
            minisimulator.LinearRange(10, 5)

        with pytest.raises(ValueError, match="stop must be greater than start"):
            minisimulator.LinearRange(10, 10)

        with pytest.raises(ValueError, match="points must be at least 2"):
            minisimulator.LinearRange(0, 10, 1)

    def test_linspace(self):
        # Test linspace method
        lr = minisimulator.LinearRange(0, 10, 5)
        expected = np.linspace(0, 10, 5)
        result = lr.linspace()
        assert_array_equal(result, expected)


class TestNormalDistribution:
    def test_init(self):
        nd = minisimulator.NormalDistribution(0, 1)
        assert nd.mean == 0
        assert nd.std == 1

    def test_sample_shape(self):
        # Test sample method returns correct shape
        nd = minisimulator.NormalDistribution(0, 1)
        samples = nd.sample((10,))
        assert samples.shape == (10,)

        samples = nd.sample((5, 3))
        assert samples.shape == (5, 3)

    def test_support_size(self):
        # Test support_size method
        nd = minisimulator.NormalDistribution(0, 1)
        assert nd.support_size() is None


class TestUniformDistribution:
    def test_init(self):
        ud = minisimulator.UniformDistribution(0, 10)
        assert ud.low == 0
        assert ud.high == 10

    def test_sample_shape(self):
        # Test sample method returns correct shape
        ud = minisimulator.UniformDistribution(0, 10)
        samples = ud.sample((10,))
        assert samples.shape == (10,)

        samples = ud.sample((5, 3))
        assert samples.shape == (5, 3)

    def test_sample_range(self):
        # Test samples are within the specified range
        ud = minisimulator.UniformDistribution(5, 15)
        samples = ud.sample((1000,))
        assert np.all(samples >= 5)
        assert np.all(samples < 15)

    def test_support_size(self):
        # Test support_size method
        ud = minisimulator.UniformDistribution(0, 10)
        assert ud.support_size() is None


class TestDiscreteDistribution:
    def test_init(self):
        dd = minisimulator.DiscreteDistribution([1, 2, 3], [0.2, 0.3, 0.5])
        assert dd.choices == [1, 2, 3]
        assert dd.probabilities == [0.2, 0.3, 0.5]

    def test_with_uniform_probabilities(self):
        dd = minisimulator.DiscreteDistribution.with_uniform_probabilities([1, 2, 3])
        assert dd.choices == [1, 2, 3]
        assert dd.probabilities == [1 / 3, 1 / 3, 1 / 3]

    def test_with_discrete_probabilities(self):
        dd = minisimulator.DiscreteDistribution.with_discrete_probabilities([(1, 0.2), (2, 0.3), (3, 0.5)])
        assert dd.choices == [1, 2, 3]
        assert dd.probabilities == [0.2, 0.3, 0.5]

    def test_sample_shape(self):
        dd = minisimulator.DiscreteDistribution([1, 2, 3], [0.2, 0.3, 0.5])
        samples = dd.sample((10,))
        assert samples.shape == (10,)

        samples = dd.sample((5, 3))
        assert samples.shape == (5, 3)

    def test_sample_values(self):
        # Test that samples contain only values from choices
        dd = minisimulator.DiscreteDistribution([1, 2, 3], [0.2, 0.3, 0.5])
        samples = dd.sample((1000,))
        assert set(np.unique(samples)).issubset({1, 2, 3})

    def test_support_size(self):
        dd = minisimulator.DiscreteDistribution([1, 2, 3], [0.2, 0.3, 0.5])
        assert dd.support_size() == 3


class TestConstantDistribution:
    def test_init(self):
        # Test with different types
        cd_float = minisimulator.ConstantDistribution(5.0)
        assert cd_float.value == 5.0

        cd_int = minisimulator.ConstantDistribution(5)
        assert cd_int.value == 5

        cd_bool = minisimulator.ConstantDistribution(True)
        assert cd_bool.value is True

    def test_sample(self):
        cd = minisimulator.ConstantDistribution(5.0)
        samples = cd.sample((10,))
        assert_array_equal(samples, np.full((10,), 5.0))

        samples = cd.sample((3, 4))
        assert_array_equal(samples, np.full((3, 4), 5.0))

    def test_support_size(self):
        cd = minisimulator.ConstantDistribution(5.0)
        assert cd.support_size() == 1


class TestModelArgumentsBuilder:
    def setup_method(self):
        # Create mock nodes for testing
        self.node1 = graph.Node("node1")
        self.node2 = graph.Node("node2")
        self.node3 = graph.Node("node3")
        self.node4 = graph.Node("node4")

    def test_add_linear_range(self):
        builder = minisimulator.ModelArgumentsBuilder()
        lr = minisimulator.LinearRange(0, 10, 5)
        builder.add(self.node1, lr)
        assert builder.params[self.node1] == lr

    def test_add_distribution(self):
        builder = minisimulator.ModelArgumentsBuilder()
        nd = minisimulator.NormalDistribution(0, 1)
        builder.add(self.node1, nd)
        assert builder.params[self.node1] == nd

    def test_build_with_linear_range(self):
        builder = minisimulator.ModelArgumentsBuilder()
        lr = minisimulator.LinearRange(0, 10, 5)
        builder.add(self.node1, lr)

        result = builder.build(10)
        assert self.node1 in result
        assert_array_equal(result[self.node1], np.linspace(0, 10, 5))

    def test_build_with_distribution(self):
        # Using a constant distribution for predictable results in tests
        builder = minisimulator.ModelArgumentsBuilder()
        cd = minisimulator.ConstantDistribution(5.0)
        builder.add(self.node1, cd)

        result = builder.build(10)
        assert self.node1 in result
        assert_array_equal(result[self.node1], np.full((10,), 5.0))

    def test_build_with_mixed_parameters(self):
        builder = minisimulator.ModelArgumentsBuilder()

        # Add a linear range
        lr = minisimulator.LinearRange(0, 10, 5)
        builder.add(self.node1, lr)

        # Add different distributions
        cd = minisimulator.ConstantDistribution(5.0)
        builder.add(self.node2, cd)

        dd = minisimulator.DiscreteDistribution.with_uniform_probabilities([1, 2, 3])
        builder.add(self.node3, dd)

        ud = minisimulator.UniformDistribution(0, 1)
        builder.add(self.node4, ud)

        # Build with ensemble size 20
        result = builder.build(20)

        # Check each parameter
        assert self.node1 in result
        assert_array_equal(result[self.node1], np.linspace(0, 10, 5))

        assert self.node2 in result
        assert_array_equal(result[self.node2], np.full((20,), 5.0))

        assert self.node3 in result
        assert result[self.node3].shape == (20,)
        assert set(np.unique(result[self.node3])).issubset({1, 2, 3})

        assert self.node4 in result
        assert result[self.node4].shape == (20,)
        assert np.all(result[self.node4] >= 0)
        assert np.all(result[self.node4] < 1)


# Test that the Distribution protocol is working correctly
def test_distribution_protocol():
    # Create instances of each distribution
    nd = minisimulator.NormalDistribution(0, 1)
    ud = minisimulator.UniformDistribution(0, 10)
    dd = minisimulator.DiscreteDistribution([1, 2, 3], [0.2, 0.3, 0.5])
    cd = minisimulator.ConstantDistribution(5.0)

    # Check that they satisfy the protocol
    assert isinstance(nd, minisimulator.Distribution)
    assert isinstance(ud, minisimulator.Distribution)
    assert isinstance(dd, minisimulator.Distribution)
    assert isinstance(cd, minisimulator.Distribution)

    # Test the common interface methods
    assert nd.sample((5,)).shape == (5,)
    assert ud.sample((5,)).shape == (5,)
    assert dd.sample((5,)).shape == (5,)
    assert cd.sample((5,)).shape == (5,)

    assert nd.support_size() is None
    assert ud.support_size() is None
    assert dd.support_size() == 3
    assert cd.support_size() == 1
