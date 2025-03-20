"""
Automatic Naming
================

Provides mechanisms to automatically name tensors for debugging purposes,
including both a context manager and a decorator.

Basic Usage
-----------

Using the context manager:

    >>> from yakof.frontend import abstract, autonaming
    >>> space = abstract.TensorSpace(abstract.bases.X)
    >>> with autonaming.context():
    ...     x = space.placeholder("")  # named 'x'
    ...     y = space.placeholder("")  # named 'y'
    ...     z = x + y                  # named 'z'

Using the decorator with class attributes:

    >>> from yakof.frontend import abstract, autonaming
    >>> class Model:
    ...     @autonaming.decorator
    ...     def __init__(self):
    ...         self.x = space.placeholder("")  # named 'x'
    ...         self.y = space.placeholder("")  # named 'y'
    ...         self.z = self.x + self.y        # named 'z'

Implementation Notes
--------------------

The context manager tracks variable assignments in the current scope and automatically
names unnamed objects that implement the Nameable protocol.

The decorator tracks attribute assignments to class instances and automatically
names unnamed objects that implement the Nameable protocol.

Supported Patterns
~~~~~~~~~~~~~~~~~~

1. Direct assignments in the context frame:

    >>> with autonaming.context():
    ...     x = space.placeholder("")  # named 'x'
    ...     y = make_placeholder("")   # named 'y'

2. Operations on named tensors:

    >>> with autonaming.context():
    ...     x = space.placeholder("")  # named 'x'
    ...     y = x + 1                  # named 'y'

3. Class attribute assignments with the decorator:

    >>> class Model:
    ...     @autonaming.decorator
    ...     def __init__(self):
    ...         self.x = space.placeholder("")  # named 'x'
    ...         self.result = self.x * 2        # named 'result'

Unsupported Patterns
~~~~~~~~~~~~~~~~~~~~

1. Objects created in comprehensions:

    >>> with autonaming.context():
    ...     tensors = [space.placeholder("") for _ in range(3)]  # NOT named
    ...     x = tensors[0]  # named 'x'

2. Objects created in nested scopes:

    >>> with autonaming.context():
    ...     def make_tensor():
    ...         return space.placeholder("")  # NOT named
    ...     x = make_tensor()  # named 'x'

Aliasing and Name Conflicts
---------------------------

When the same tensor is assigned to multiple names:

    >>> with autonaming.context():
    ...     x = space.placeholder("")  # Gets named 'x'
    ...     y = x                      # Warning: attempting to rename

The mechanisms:
1. Use the first name they encounter for the tensor
2. Emit a warning for subsequent naming attempts
3. Suggest adjusting code to avoid aliasing

Note: The order in which names are processed depends on Python's dictionary
iteration order. While this is stable within a Python version since 3.7,
it means the "first" name might not be the one that appears first in your
code. For reliable debugging, avoid assigning multiple names to the same
tensor.

Best Practices
--------------

DO:
- Use meaningful, unique names for tensors
- Use the context manager for local variables
- Use the decorator for class attributes
- Use the context manager for debugging purposes

DON'T:
- Rely on specific names/strings for the model logic
- Assign the same tensor to multiple names
- Create tensors in comprehensions or nested functions
"""

import inspect
import logging
import types
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Nameable(Protocol):
    """
    Protocol for objects with name setters and getters.

    Methods:
        implements_namer: adds extra methods to the expected interface to avoid
        name setting for objects that do implement name getter/setter but are
        not designed to work along with the autonaming functionality.

        name: getter for the name of the object.

        name: setter for the name of the object.
    """

    def implements_namer(self) -> None: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, value: str) -> None: ...


# TODO(bassosimone): consider taking advantage of the `graph.Node.id`
# field to walk through the nameables respecting the order in which they
# were defined in the code, which makes renames possible.


def _maybe_autoname(var: Any, name: str) -> None:
    if not isinstance(var, Nameable):
        return
    if var.name and var.name != name:
        logging.warning(f"autonaming: attempting to rename {var.name} to {name}")
        logging.warning(f"autonaming: debugging code will use {var.name}")
        logging.warning("autonaming: adjust your code to avoid aliasing tensors")
        return
    var.name = name


class context:
    """
    Context manager for automatically assigning names to Nameable objects.

    Assigning multiple names to the same tensor, such as in the following example:

        >>> with autonaming.context():
        ...     x = space.placeholder("")
        ...     y = x

    will cause one of the two names at ~random to be used when debugging. There
    is no need to assign multiple names to the same underlying tensor. If you use
    this coding style, the context will emit a warning, to let you know you are
    making your own debugging more complicated than needed.
    """

    def __enter__(self) -> None:
        cf = inspect.currentframe()
        assert cf is not None
        frame = cf.f_back
        assert frame is not None
        self._initial_locals = set(frame.f_locals.keys())

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        if exc_type is not None:
            return  # do nothing if there was an exception
        cf = inspect.currentframe()
        assert cf is not None
        frame = cf.f_back
        assert frame is not None
        new_vars = set(frame.f_locals.keys()) - self._initial_locals
        for name in new_vars:
            var = frame.f_locals[name]
            _maybe_autoname(var, name)


def decorator(method):
    """
    Decorator that automatically names Nameable attributes in methods.

    This is particularly useful for naming class attributes created in __init__
    methods, where the context manager cannot detect attribute assignments.

    Example:
        >>> class Model:
        ...     @autonaming.decorator
        ...     def __init__(self):
        ...         self.x = space.placeholder("")  # named 'x'
        ...         self.y = space.placeholder("")  # named 'y'
    """

    def wrapper(self, *args, **kwargs):
        # Get initial attributes
        initial_attrs = set(vars(self).keys() if hasattr(self, "__dict__") else [])

        # Call the original method
        result = method(self, *args, **kwargs)

        # Find new attributes
        new_attrs = set(vars(self).keys() if hasattr(self, "__dict__") else []) - initial_attrs

        # Apply naming to new attributes
        for attr_name in new_attrs:
            attr_value = getattr(self, attr_name)
            _maybe_autoname(attr_value, attr_name)

        return result

    return wrapper
