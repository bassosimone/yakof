"""
Automatic Naming
================

Provides a context manager to automatically name tensors.

Example
-------

In the following example:

    >>> from yakof.frontend import abstract, autonaming
    >>> space = abstract.TensorSpace(abstract.bases.X)
    >>> with autonaming.context():
    ...     x = space.placeholder()
    ...     y = space.placeholder()
    ...     z = x + y

the tensors will be automatically named x, y, and z.
"""

from typing import Protocol, runtime_checkable

import inspect
import types


@runtime_checkable
class Nameable(Protocol):
    """Protocol for objects with name setters and getters."""

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, value: str) -> None: ...


class context:
    """Context manager for automatically assigning names to Nameable objects."""

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
            if isinstance(var, Nameable) and not var.name:
                var.name = name
