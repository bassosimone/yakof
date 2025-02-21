"""
Automatic Naming
================

Provides a context manager to automatically name tensors for debugging purposes.

Basic Usage
-----------

Here's a simple example:

    >>> from yakof.frontend import abstract, autonaming
    >>> space = abstract.TensorSpace(abstract.bases.X)
    >>> with autonaming.context():
    ...     x = space.placeholder()  # named 'x'
    ...     y = space.placeholder()  # named 'y'
    ...     z = x + y                # named 'z'

Implementation Notes
------------------

The context manager tracks variable assignments in the current scope and automatically
names unnamed objects that implement the Nameable protocol.

Supported Patterns
~~~~~~~~~~~~~~~~~~

1. Direct assignments in the context frame:

    >>> with autonaming.context():
    ...     x = space.placeholder()  # named 'x'
    ...     y = make_placeholder()   # named 'y'

2. Operations on named tensors:

    >>> with autonaming.context():
    ...     x = space.placeholder()  # named 'x'
    ...     y = x + 1               # named 'y'

Unsupported Patterns
~~~~~~~~~~~~~~~~~~~~

1. Objects created in comprehensions:

    >>> with autonaming.context():
    ...     tensors = [space.placeholder() for _ in range(3)]  # NOT named
    ...     x = tensors[0]  # named 'x'

2. Objects created in nested scopes:

    >>> with autonaming.context():
    ...     def make_tensor():
    ...         return space.placeholder()  # NOT named
    ...     x = make_tensor()  # named 'x'

Aliasing and Name Conflicts
---------------------------

When the same tensor is assigned to multiple names:

    >>> with autonaming.context():
    ...     x = space.placeholder()  # Gets named 'x'
    ...     y = x                    # Warning: attempting to rename

The context manager:
1. Uses the first name it encounters for the tensor
2. Emits a warning for subsequent naming attempts
3. Suggests adjusting code to avoid aliasing

Note: The order in which names are processed depends on Python's dictionary
iteration order. While this is stable within a Python version since 3.7,
it means the "first" name might not be the one that appears first in your
code. For reliable debugging, avoid assigning multiple names to the same
tensor.

Best Practices
-------------

DO:
- Use meaningful, unique names for tensors
- Use the context manager for debugging purposes

DON'T:
- Rely on specific names/strings for the model logic
- Assign the same tensor to multiple names
- Create tensors in comprehensions or nested functions
"""

from typing import Protocol, runtime_checkable

import inspect
import logging
import types


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


class context:
    """
    Context manager for automatically assigning names to Nameable objects.

    Assigning multiple names to the same tensor, such as in the following example:

        >>> with autonaming.context():
        ...     x = space.placeholder()
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
            if not isinstance(var, Nameable):
                continue
            if var.name:
                logging.warning(
                    f"autonaming: attempting to rename {var.name} to {name}"
                )
                logging.warning(f"autonaming: debugging code will use {var.name}")
                logging.warning(
                    f"autonaming: consider adjusting your code to avoid aliasing tensors"
                )
                continue
            var.name = name
