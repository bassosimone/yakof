"""
Tensor Language Frontend
========================

This package provides a type-safe language for building tensor computations.
It consists of two main components:

1. An abstract syntax tree (AST) for representing computations (graph.py)
2. A typed tensor language built on top of the AST (abstract.py)

The frontend allows you to:
- Build computation graphs with type safety
- Work with tensors in different spaces
- Transform tensors between spaces using morphisms

Architecture Decisions
----------------------

1. Separation of Concerns:
   - Graph building (graph.py)
   - Type safety (abstract.py)
   - Basis definitions (bases.py)

2. Progressive Lowering:
   - Abstract tensor operations
   - Concrete computation graph
   - Backend-specific IR
   - Linear form
   - VM execution

3. Type Safety:
   - Generic types for basis vectors
   - Compile-time checking of operations
   - Safe transformations via morphisms

This layered approach enables:
- Clear separation of concerns
- Type safety throughout compilation
- Multiple backend support
- Future optimizations at each level
"""

# SPDX-License-Identifier: Apache-2.0
