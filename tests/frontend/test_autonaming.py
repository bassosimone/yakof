"""Tests for the yakof.frontend.autonaming module."""

# SPDX-License-Identifier: Apache-2.0

import logging

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
        _ = "not_an_object"
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
    with autonaming.context():
        y = x
    assert x.name == "x"
    assert y.name == "x"


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
    assert "adjust your code" in caplog.records[2].message
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
            _ = obj
            raise ValueError("test error")
    except ValueError:
        pass

    assert obj.name == ""  # Should not be named due to exception


def test_decorator_basic():
    """Test basic decorator functionality."""

    class TestClass:
        @autonaming.decorator
        def __init__(self):
            self.x = NameableObject()
            self.y = NameableObject()

    instance = TestClass()
    assert instance.x.name == "x"
    assert instance.y.name == "y"


def test_decorator_respect_existing_names():
    """Test decorator respects existing names."""

    class TestClass:
        @autonaming.decorator
        def __init__(self):
            self.x = NameableObject("existing")
            self.y = NameableObject()

    instance = TestClass()
    assert instance.x.name == "existing"  # Preserved
    assert instance.y.name == "y"  # Auto-named


def test_decorator_non_nameable_objects():
    """Test decorator correctly handles non-Nameable objects."""

    class TestClass:
        @autonaming.decorator
        def __init__(self):
            self.x = NameableObject()
            self.y = HasNameButNotNameable()
            self.z = "not_an_object"

    instance = TestClass()
    assert instance.x.name == "x"  # Should be named
    assert instance.y.name == ""  # Should be untouched


def test_decorator_nested_methods():
    """Test decorator with nested method calls."""

    class TestClass:
        @autonaming.decorator
        def __init__(self):
            self.x = NameableObject()
            self._make_other_object()

        def _make_other_object(self):
            self.y = NameableObject()  # Should still be named

    instance = TestClass()
    assert instance.x.name == "x"
    assert instance.y.name == "y"


def test_decorator_inheritance():
    """Test decorator with class inheritance."""

    class BaseClass:
        @autonaming.decorator
        def __init__(self):
            self.base = NameableObject()

    class DerivedClass(BaseClass):
        @autonaming.decorator
        def __init__(self):
            super().__init__()
            self.derived = NameableObject()

    instance = DerivedClass()
    assert instance.base.name == "base"
    assert instance.derived.name == "derived"


def test_decorator_complex_containers():
    """Test decorator with complex data structures."""

    class TestClass:
        @autonaming.decorator
        def __init__(self):
            self.x = NameableObject()
            self.list = [NameableObject() for _ in range(3)]
            self.dict = {"key": NameableObject()}

    instance = TestClass()
    assert instance.x.name == "x"
    assert instance.list[0].name == ""  # List elements shouldn't be auto-named
    assert instance.dict["key"].name == ""  # Dict values shouldn't be auto-named


def test_decorator_reassignment():
    """Test decorator behavior with attribute reassignment."""

    class TestClass:
        @autonaming.decorator
        def __init__(self):
            self.x = NameableObject()
            self.y = self.x  # Same object, should trigger warning

    instance = TestClass()
    # Note: test is not very specific because the order with
    # which we process names is not guaranteed
    assert instance.x.name in ("x", "y")
    assert instance.y.name == instance.x.name


def test_decorator_with_args():
    """Test decorator on methods with arguments."""

    class TestClass:
        @autonaming.decorator
        def __init__(self, arg1, arg2=None):
            self.x = NameableObject()
            self.arg1 = arg1
            self.arg2 = arg2
            if arg2:
                self.y = NameableObject()

    instance1 = TestClass("test")
    instance2 = TestClass("test", "value")

    assert instance1.x.name == "x"
    assert not hasattr(instance1, "y")

    assert instance2.x.name == "x"
    assert instance2.y.name == "y"


def test_decorator_other_methods():
    """Test decorator on non-init methods."""

    class TestClass:
        def __init__(self):
            self.initial = NameableObject()

        @autonaming.decorator
        def add_more(self):
            self.added = NameableObject()
            return self.added

    instance = TestClass()
    result = instance.add_more()

    assert instance.initial.name == ""  # Not named by decorator
    assert instance.added.name == "added"
    assert result.name == "added"  # Same object


def test_decorator_warnings(caplog):
    """Test warning behavior with decorator."""
    caplog.set_level(logging.WARNING)

    class TestClass:
        @autonaming.decorator
        def __init__(self):
            self.x = NameableObject()
            # This should trigger a warning
            self.y = self.x

    instance = TestClass()

    assert len(caplog.records) == 3
    assert "attempting to rename" in caplog.records[0].message
    assert "debugging code will use" in caplog.records[1].message
    assert "adjust your code" in caplog.records[2].message
    assert instance.x.name == instance.y.name


def test_decorator_exception_handling():
    """Test decorator behavior when exceptions occur."""
    obj = NameableObject()

    class TestClass:
        @autonaming.decorator
        def __init__(self):
            self.x = obj
            raise ValueError("test error")

    try:
        TestClass()
    except ValueError:
        pass

    assert obj.name == ""  # Should not be named due to exception


def test_decorator_attribute_deletion():
    """Test decorator with attribute deletion."""

    class TestClass:
        @autonaming.decorator
        def __init__(self):
            self.x = NameableObject()
            self.temp = NameableObject()
            delattr(self, "temp")  # Delete before method ends

    instance = TestClass()
    assert instance.x.name == "x"
    assert not hasattr(instance, "temp")


def test_decorator_with_properties():
    """Test decorator behavior with properties."""

    class TestClass:
        @autonaming.decorator
        def __init__(self):
            self._obj = NameableObject()

        @property
        def prop_obj(self):
            return self._obj

    instance = TestClass()
    assert instance._obj.name == "_obj"  # Should use attribute name, not property name
