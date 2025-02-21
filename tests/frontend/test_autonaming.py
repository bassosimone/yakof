"""Tests for the yakof.frontend.autonaming module."""

# SPDX-License-Identifier: Apache-2.0

from typing import Any

import logging
import pytest

from yakof.frontend import autonaming


class NameableObject:
    """Test class implementing the Nameable protocol."""

    def __init__(self, initial_name: str = "") -> None:
        self._name = initial_name

    def implements_namer(self) -> None:
        """Part of the autonaming.Nameable protocol."""

    @property
    def name(self) -> str:
        """Part of the autonaming.Nameable protocol."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Part of the autonaming.Nameable protocol."""
        self._name = value


class HasNameButNotNameable:
    """Test class with name attribute but not implementing Nameable protocol."""

    def __init__(self) -> None:
        self.name = ""


def test_basic_naming():
    """Test basic automatic naming functionality."""
    with autonaming.context():
        x = NameableObject()
        y = NameableObject()
    assert x.name == "x"
    assert y.name == "y"


def test_respect_existing_names():
    """Test that existing names are not overwritten."""
    with autonaming.context():
        x = NameableObject("existing")
        y = NameableObject()
    assert x.name == "existing"  # Preserved
    assert y.name == "y"  # Auto-named


def test_non_nameable_objects():
    """Test that non-Nameable objects are ignored."""
    with autonaming.context():
        x = NameableObject()
        y = HasNameButNotNameable()
        z = "not_an_object"
    assert x.name == "x"  # Should be named
    assert y.name == ""  # Should be untouched


def test_nested_contexts():
    """Test that nested naming contexts work correctly."""
    with autonaming.context():
        x = NameableObject()
        with autonaming.context():
            y = NameableObject()
    assert x.name == "x"
    assert y.name == "y"


def test_multiple_variables():
    """Test naming multiple variables in complex scenarios."""
    with autonaming.context():
        x = NameableObject()
        y = [NameableObject() for _ in range(3)]  # List of nameables
        z = {"key": NameableObject()}  # Dict with nameable
    assert x.name == "x"
    assert y[0].name == ""  # List elements shouldn't be auto-named
    assert z["key"].name == ""  # Dict values shouldn't be auto-named


def test_protocol_checking():
    """Test runtime protocol checking behavior."""
    assert isinstance(NameableObject(), autonaming.Nameable)
    assert not isinstance(HasNameButNotNameable(), autonaming.Nameable)
    assert not isinstance("string", autonaming.Nameable)


def test_reassignment():
    """Test naming behavior with variable reassignment."""
    with autonaming.context():
        x = NameableObject()
        x = NameableObject()  # Reassignment
    assert x.name == "x"  # Last assignment should be named


def test_frame_behavior():
    """Test basic frame behavior."""
    with autonaming.context():
        x = NameableObject()  # Should be named

        def nested():
            y = NameableObject()  # Should NOT be named
            return y

        z = nested()  # Should be named when assigned here

    assert x.name == "x"
    assert z.name == "z"


def test_naming_warnings(caplog):
    """Test warning behavior for multiple names."""
    caplog.set_level(logging.WARNING)

    with autonaming.context():
        x = NameableObject()
        y = x  # Should trigger warning

    assert len(caplog.records) == 3
    assert "attempting to rename" in caplog.records[0].message
    assert "debugging code will use" in caplog.records[1].message
    assert "consider adjusting your code" in caplog.records[2].message
    assert x.name == y.name  # Same object has same name


def test_comprehension_edge_cases():
    """Test basic comprehension behavior."""
    with autonaming.context():
        xs = [NameableObject() for _ in range(2)]
        first = xs[0]  # Should be named

    assert first.name == "first"
    assert xs[1].name == ""  # Not directly named


def test_exception_handling():
    """Test naming behavior when exceptions occur."""
    obj = NameableObject()

    try:
        with autonaming.context():
            x = obj
            raise ValueError("test error")
    except ValueError:
        pass

    assert obj.name == ""  # Should not be named due to exception
