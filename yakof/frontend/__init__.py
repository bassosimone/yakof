"""
Tensor Language Frontend
========================

This package provides a type-safe, internal domain-specific language
for building tensor computations. It consists of two components:

1. An abstract syntax tree (AST) for representing computations (graph.py)
2. A typed tensor language built on top of the AST (abstract.py)

The `frontend` package allows you to:

- Build computation graphs with type safety
- Work with tensors in different spaces
- Transform tensors between spaces using tensor maps

On mathematical terminology
---------------------------

This package uses 'tensor' in the computational sense (i.e., multidimensional
arrays) while borrowing mathematical concepts like bases and vector spaces
to provide a structured way to handle transformations between different
dimensional spaces.

While not strictly adhering to mathematical tensor theory, this approach
provides a practical framework for engineering computations.

Architecture Decisions
----------------------

1. Separation of Concerns:
   - Type safety (abstract.py)
   - Basis definitions (bases.py)
   - Graph building (graph.py)

2. Progressive Lowering:
   - Abstract tensor operations
   - Computation graph
   - Backend-specific code generation (e.g., `numpybackend` package)

3. Type Safety:
   - Generic types for basis vectors
   - Compile-time checking of tensor operations
   - Space transformations via tensor maps

This layered approach enables:
- Separation of concerns
- Type safety throughout compilation
- Multiple backend support
- Future optimizations at each level

See Also
--------

yakof.frontend.abstract
    Abstract tensor operations built on top of yakof.frontend.graph.

yakof.frontend.bases
    Basis definitions for tensor spaces to be used along with yakof.frontend.abstract.

yakof.frontend.graph
    Computation graph building.

Example
-------

The following example creates a tensor space, defines two placeholder
tensors, and symbolically computes their sum:

    >>> from yakof.frontend import abstract, bases
    >>>
    >>> space = abstract.TensorSpace(bases.X)
    >>>
    >>> a = space.placeholder("x")
    >>> b = space.placeholder("c")
    >>> c = a + b

The computation graph can be transformed into a backend-specific
representation for evaluation using, e.g., yakof.numpybackend.
"""

# SPDX-License-Identifier: Apache-2.0
