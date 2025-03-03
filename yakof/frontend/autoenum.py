"""
Enumeration Support
===================

This module contains code to generate automatically disjoint type-safe
enumerations that can be integrated with tensor spaces.

The module provides these abstractions:

1. Type: an enumeration type associated with a tensor space. Each enum type
has a disjoint range of integer values from other enum types.

2. Value:Represents a specific value within an enumeration type. Each
value has a unique integer representation and an associated tensor.

We reserve 20 bits for the enumeration type ID and 20 bits for the value ID,
which should provide plenty of enumeration space. When we're out of the
enumeration space (i.e., we have used 20 bits), the code throws a ValueError.

Type Parameters
---------------

E: The basis type of the tensor space associated with an enumeration.

Usage Example
-------------

```python
from yakof.frontend import abstract, autoenum, autonaming

# Define spaces and enum types
with autonaming.context():
    weather_enum = autoenum.Type(weather_space, "")
    SUNNY = autoenum.Value(weather_enum, "")
    CLOUDY = autoenum.Value(weather_enum, "")

    time_enum = autoenum.Type(time_space, "")
    MORNING = autoenum.Value(time_enum, "")
    EVENING = autoenum.Value(time_enum, "")

# Use in expressions (type-safe)
is_sunny = weather_tensor == SUNNY.tensor  # Valid
is_morning = time_tensor == MORNING.tensor  # Valid
# weather_tensor == MORNING.tensor  # Type error - different enum types
```
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Generic, TypeVar

from .. import atomic
from ..frontend import abstract


_id_generator = atomic.Int()
"""Atomic integer generator for unique enum type IDs."""


_shift = 8
"""Bit shift used to separate enum type IDs from value IDs within the type."""


E = TypeVar("E")
"""Type variable for tensor basis types, used by Type and Value."""


def _next_id(gen: atomic.Int) -> int:
    """Generate the next ID from an atomic counter with overflow protection.

    Args:
        gen: Atomic counter to increment for generating the next ID

    Returns:
        The next unique ID

    Raises:
        ValueError: If the counter exceeds the maximum allowed value
    """
    # TODO(bassosimone): maybe check before incrementing?
    value = gen.add(1)
    if value >= (1 << _shift):
        raise ValueError("Too many enum values")
    return value


class Type(Generic[E]):
    """A type-safe enumeration type bound to a tensor space.

    Type Parameters:
        E: The basis type of the associated tensor space.

    Attributes:
        space: The tensor space associated with this enumeration type.
        basevalue: Unique ID for this enum type.
        gen: Atomic counter for generating unique value IDs within this type.

    Args:
        space: The tensor space to associate with this enum type.
        name: name for the enum type (leave empty if using autonaming).
    """

    def __init__(self, space: abstract.TensorSpace[E], name: str) -> None:
        self.basevalue = _next_id(_id_generator) << _shift
        self.gen = atomic.Int()
        self.space = space
        self.tensor = self.space.placeholder(name=name)

    # autonaming.Namer protocol implementation
    def implements_namer(self) -> None:
        """This method is part of the autonaming.Namer protocol"""

    @property
    def name(self) -> str:
        """This method is part of the autonaming.Namer protocol"""
        return self.tensor.name

    @name.setter
    def name(self, value: str) -> None:
        """This method is part of the autonaming.Namer protocol"""
        self.tensor.name = value


class Value(Generic[E]):
    """A specific value within an enumeration type.

    Each value has a unique integer representation that:
    1. Is disjoint from values in other enum types
    2. Can be used in tensor computations via the .tensor property

    Type Parameters:
        E: The basis type of the tensor space from the parent enum type.

    Attributes:
        value: Integer representation of this enum value.
        tensor: Constant tensor representation in the parent's tensor space.

    Args:
        parent: The enum type this value belongs to.
        name: name for this value (leave empty if using autonaming).
    """

    def __init__(self, parent: Type[E], name: str) -> None:
        self._type = parent
        self.value = parent.basevalue | _next_id(parent.gen)
        self.tensor = parent.space.constant(self.value, name=name)

    # autonaming.Namer protocol implementation
    def implements_namer(self) -> None:
        """This method is part of the autonaming.Namer protocol"""

    @property
    def name(self) -> str:
        """This method is part of the autonaming.Namer protocol"""
        return self.tensor.name

    @name.setter
    def name(self, value: str) -> None:
        """This method is part of the autonaming.Namer protocol"""
        self.tensor.name = value
