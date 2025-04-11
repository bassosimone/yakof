"""
Tensor Language Frontend.
========================

This package provides a type-safe, internal domain-specific language
for building tensor computations.

It builds on top of the `dt-model` graph library. The overall architecture
supports the following components:

1. Abstract Syntax Tree (graph.py)
   - Core computation graph representation
   - Node types for operations (add, multiply, where, etc.)
   - Debug facilities (tracepoints, breakpoints)

2. Typed Tensor Language (abstract.py)
   - Type-safe tensors with mathematical operations
   - Generic types for basis/space safety
   - Tensor spaces for structured computation

3. Morphisms Between Spaces (morphisms.py)
   - Type-safe transformations between tensor spaces
   - Space expansion (adding dimensions)
   - Projection operations (dimension reduction)

4. Canonical Bases and Spaces (bases.py, spaces.py)
   - Pre-defined tensor spaces from R⁰ to R⁶
   - Named basis vectors (X, Y, Z, U, V, W)
   - Ready-to-use expansion and projection morphisms

5. Debugging and Visualization Tools
   - Automatic naming of tensors (autonaming.py)
   - Computation graph linearization (linearize.py)
   - Pretty printing of expressions (pretty.py)

6. Type-Safe Enumerations (autoenum.py)
   - Integrated with tensor spaces
   - Disjoint enumeration types
   - Automatic value generation

Usage Example
-------------

```python
from yakof.frontend import abstract, spaces, autonaming

# Create computation with automatic naming
with autonaming.context():
    # Create tensors in a space
    a = spaces.x.placeholder("")  # Named 'a'
    b = spaces.x.placeholder("")  # Named 'b'

    # Perform operations
    c = a + b                     # Named 'c'
    d = c * 2                     # Named 'd'

    # Transform between spaces
    xy_d = spaces.expand_x_to_xy(d)  # Expand to 2D
```

Mathematical Framework
----------------------

The package uses mathematical concepts like bases and tensor spaces to provide:

- Type safety through generic types
- Compile-time checking of tensor operations
- Safe transformations between spaces of different dimensions

This approach enables a clean separation between:
- Abstract operations (add, multiply)
- Space-specific behavior (projections, expansions)
- Execution model (graph representation)

Design Principles
-----------------

1. Composability
   - Small, focused components that work together
   - Category theory-inspired design for transformations

2. Type Safety
   - Prevents mixing tensors from incompatible spaces
   - Provides clear error messages at compile time

3. Progressive Lowering
   - High-level tensor operations
   - Mid-level computation graph
   - Low-level backend-specific code

See Also
--------
- yakof.frontend.abstract: Type-safe tensor operations
- yakof.frontend.bases: Basis definitions for tensor spaces
- yakof.frontend.spaces: Pre-defined tensor spaces and transformations
- yakof.frontend.morphisms: Transformations between tensor spaces
- yakof.frontend.graph: Low-level computation graph building
- yakof.frontend.autonaming: Automatic tensor naming for debugging
- yakof.frontend.autoenum: Type-safe enumeration support
- yakof.frontend.linearize: Topological sorting of computation graphs
- yakof.frontend.pretty: String formatting for computation graphs
"""

# SPDX-License-Identifier: Apache-2.0
