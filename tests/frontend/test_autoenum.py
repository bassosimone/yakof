"""Tests for the yakof.frontend.autoenum module."""

# SPDX-License-Identifier: Apache-2.0

import pytest

from yakof import atomic
from yakof.frontend import abstract, autoenum, autonaming, bases, graph


# Define test bases and spaces
class TestBasis:
    axes = (1000,)


class OtherBasis:
    axes = (2000,)


@pytest.fixture
def test_space():
    return abstract.TensorSpace(TestBasis())


@pytest.fixture
def other_space():
    return abstract.TensorSpace(OtherBasis())


def test_enum_type_creation(test_space):
    """Test creation of enum types."""
    enum_type = autoenum.Type(test_space, "TestEnum")

    assert enum_type.name == "TestEnum"
    assert enum_type.space is test_space
    assert enum_type.basevalue > 0
    assert enum_type.basevalue % (1 << autoenum.BITS_PER_ENUM_SPACE) == 0  # Should be left-shifted


def test_enum_value_creation(test_space):
    """Test creation of enum values."""
    enum_type = autoenum.Type(test_space, "TestEnum")
    enum_value = autoenum.Value(enum_type, "TestValue")

    assert enum_value.name == "TestValue"
    assert (
        enum_value.value & (enum_type.basevalue) == enum_type.basevalue
    )  # Contains type's base value
    assert isinstance(enum_value.tensor, abstract.Tensor)
    assert enum_value.tensor.space is test_space


def test_enum_value_uniqueness(test_space):
    """Test that enum values in the same type have unique values."""
    enum_type = autoenum.Type(test_space, "TestEnum")
    value1 = autoenum.Value(enum_type, "Value1")
    value2 = autoenum.Value(enum_type, "Value2")

    assert value1.value != value2.value
    assert isinstance(value1.tensor.node, graph.constant)
    assert isinstance(value2.tensor.node, graph.constant)
    assert value1.tensor.node.value != value2.tensor.node.value


def test_enum_type_disjoint_ranges(test_space, other_space):
    """Test that different enum types have disjoint value ranges."""
    type1 = autoenum.Type(test_space, "Type1")
    type2 = autoenum.Type(other_space, "Type2")

    value1 = autoenum.Value(type1, "Value1")
    value2 = autoenum.Value(type2, "Value2")

    # Values from different enum types should have different ranges
    assert (value1.value & type1.basevalue) != (value2.value & type2.basevalue)


def test_autonaming_integration():
    """Test integration with autonaming context."""
    test_space = abstract.TensorSpace(TestBasis())

    with autonaming.context():
        weather_enum = autoenum.Type(test_space, "")
        sunny = autoenum.Value(weather_enum, "")
        cloudy = autoenum.Value(weather_enum, "")

    assert weather_enum.name == "weather_enum"
    assert sunny.name == "sunny"
    assert cloudy.name == "cloudy"

    # Tensor names should also be set
    assert sunny.tensor.name == "sunny"
    assert cloudy.tensor.name == "cloudy"


def test_tensor_comparison(test_space):
    """Test using enum values in tensor comparisons."""
    enum_type = autoenum.Type(test_space, "TestEnum")
    value1 = autoenum.Value(enum_type, "Value1")
    value2 = autoenum.Value(enum_type, "Value2")

    # Create a placeholder tensor in the same space
    tensor = test_space.placeholder("test_tensor")

    # Test tensor comparisons
    eq_tensor1 = tensor == value1.tensor
    eq_tensor2 = tensor == value2.tensor

    # Verify the comparisons created equality operations
    assert eq_tensor1.node.__class__.__name__ == "equal"
    assert eq_tensor2.node.__class__.__name__ == "equal"

    # Verify the correct values are being compared
    assert eq_tensor1.node.right.value == value1.value
    assert eq_tensor2.node.right.value == value2.value


def test_many_enum_values():
    """Test creating many enum values (not enough to hit the limit)."""
    test_space = abstract.TensorSpace(TestBasis())
    enum_type = autoenum.Type(test_space, "LargeEnum")

    # Create a moderate number of values
    values = [autoenum.Value(enum_type, f"Value{i}") for i in range(100)]

    # Check they're all unique
    unique_values = {v.value for v in values}
    assert len(unique_values) == 100


def test_value_id_overflow():
    """Test that an error is raised when too many enum values are created."""
    test_space = abstract.TensorSpace(TestBasis())
    enum_type = autoenum.Type(test_space, "OverflowTest")

    # Manually set the counter to near the limit
    enum_type.gen.add((1 << autoenum.BITS_PER_ENUM_SPACE) - 3)

    # Create two values (should work)
    value1 = autoenum.Value(enum_type, "Value1")
    value2 = autoenum.Value(enum_type, "Value2")

    # Next one should raise an error
    with pytest.raises(ValueError, match="Too many enum values"):
        value3 = autoenum.Value(enum_type, "Value3")


def test_id_generation():
    """Test the _next_id helper function directly."""
    counter = atomic.Int()

    # Generate a few IDs
    id1 = autoenum._next_id(counter)
    id2 = autoenum._next_id(counter)
    id3 = autoenum._next_id(counter)

    assert id1 == 1
    assert id2 == 2
    assert id3 == 3

    # Test overflow
    counter.add((1 << autoenum.BITS_PER_ENUM_SPACE) - 3)
    with pytest.raises(ValueError):
        autoenum._next_id(counter)


def test_type_parameter_consistency():
    """Test that the generic type parameter correctly enforces type safety."""
    # This test is primarily a compile-time check
    # Here we're just verifying the attributes to ensure the code paths work

    test_space = abstract.TensorSpace(TestBasis())
    other_space = abstract.TensorSpace(OtherBasis())

    test_enum = autoenum.Type(test_space, "TestEnum")
    other_enum = autoenum.Type(other_space, "OtherEnum")

    test_value = autoenum.Value(test_enum, "TestValue")
    other_value = autoenum.Value(other_enum, "OtherValue")

    # Verify space relationships
    assert test_value.tensor.space is test_space
    assert other_value.tensor.space is other_space

    # The following would be type errors if actually attempted:
    # bad_value = autoenum.Value[TestBasis](other_enum, "BadValue")
    # test_tensor == other_value.tensor  # Would fail at compile-time


def test_enum_type_tensor_placeholder(test_space):
    """Test that enum types have a properly initialized placeholder tensor."""
    enum_type = autoenum.Type(test_space, "TestEnum")

    # Verify the tensor is initialized correctly
    assert enum_type.tensor is not None
    assert enum_type.tensor.space is test_space
    assert isinstance(enum_type.tensor, abstract.Tensor)
    assert isinstance(enum_type.tensor.node, graph.placeholder)
    assert enum_type.tensor.name == "TestEnum"

    # Verify we can use it in comparisons/operations
    test_tensor = test_space.placeholder("test")
    eq_result = test_tensor == enum_type.tensor
    assert eq_result.node.__class__.__name__ == "equal"


def test_enum_type_placeholder_autonaming():
    """Test that enum type placeholder tensors work with autonaming."""
    test_space = abstract.TensorSpace(TestBasis())

    with autonaming.context():
        weather_enum = autoenum.Type(test_space, "")

    # The placeholder tensor should have the same name as the enum type
    assert weather_enum.name == "weather_enum"
    assert weather_enum.tensor.name == "weather_enum"

    # Verify we can change the name and it propagates
    weather_enum.name = "climate_enum"
    assert weather_enum.name == "climate_enum"
    assert weather_enum.tensor.name == "climate_enum"
