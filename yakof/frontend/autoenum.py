"""
Enumeration Support
===================

This module contains code to generate automatically
disjoint type-safe enumerations.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Final
from enum import auto, Enum


# TODO(bassosimone): finish sketching out and integrating this
# code into the rest of the frontend code.


@dataclass(frozen=True)
class Value:
    """A strongly typed enum value with guaranteed
    unique integer representation."""

    enum_name: str  # The enum this value belongs to
    name: str  # The value's name
    value: int  # The unique integer value

    def __eq__(self, other) -> bool:
        if not isinstance(other, Value):
            return NotImplemented
        return self.value == other.value


class Space:
    """A space for disjoint enumerations."""

    def __init__(self) -> None:
        self._next_enum_id: int = 0
        self._names = set()

    def define_enum[T](self, name: str, *values: str) -> tuple[Value, ...]:
        """Define a new enum with the given values.

        Returns a tuple of EnumValue objects with guaranteed disjoint values.
        """
        if name in self._names:
            raise ValueError(f"Enum {name} already defined")

        # Validate names
        for value in values:
            if not value.isidentifier():
                raise ValueError(f"Invalid enum value name: {value}")
        if len(set(values)) != len(values):
            seen = set()
            dupes = [v for v in values if v in seen or seen.add(v)]
            raise ValueError(f"Duplicate enum values: {dupes}")

        # Calculate base value for this enum
        base = self._next_enum_id << 16
        self._next_enum_id += 1

        # Create enum values
        enum_values = tuple(
            Value(name, value, base + i) for i, value in enumerate(values)
        )
        self._names.add(name)
        return enum_values
