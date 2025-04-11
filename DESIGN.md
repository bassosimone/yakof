# YAKOF (Yet Another Kettle of Fish)

## Overview

YAKOF is a technology demonstrator showing how we could evolve the existing
[dt-model](https://github.com/fbk-most/dt-model) package. The key
improvements demonstrated are:

1. Type-safe tensor spaces with enforced dimensional semantics
2. Category-inspired morphisms between tensor spaces

## Design Philosophy

This project follows these design principles:

- Favor simple, composable components over complex, monolithic solutions
- Handle the common case well and let exceptional cases fail naturally
- Put complexity in the type system, not in the runtime code
- Make debugging an integrated part of the design, not an afterthought

## Core Design Decisions

### 1. Type-Safe Tensor Spaces

**Problem**: The original implementation required careful management of:
- Axis semantics (what each dimension means)
- Shape compatibility in operations
- Broadcasting rules across different kinds of parameters

**Solution**: Type-safe tensor spaces (in `frontend/abstract.py`):
- Encode tensor dimension semantics in the type system
- Use generics to enforce operations only between compatible spaces
- Raise compile-time errors for dimension mismatches

Tensor spaces are parameterized by their basis, ensuring that operations
like addition only happen between tensors from the same space.

### 2. Category-Inspired Morphisms

**Problem**: In sustainability modeling, tensors exist in different spaces:
- Time series (1D)
- Ensemble variations (1D)
- Field combinations (multi-dimensional)

Transforming between these spaces manually is error-prone.

**Solution**: Space morphisms (in `frontend/morphisms.py`):
- Provide explicit operations to transform between tensor spaces
- Enforce correct dimensional semantics
- Implement canonical axis ordering and transformation

Key morphisms include `ExpandDims` to add dimensions and
`ProjectUsingSum` to reduce dimensions while preserving semantics.

## Architecture

The architecture is composed of focused modules:

```
yakof/
│
├── frontend/          # Core computational abstractions
│   ├── abstract.py    # Tensor spaces and tensors
│   ├── autoenum.py    # Type-safe enumeration support
│   ├── autonaming.py  # Automatic naming utilities
│   ├── bases.py       # Canonical tensor bases
│   ├── morphisms.py   # Transformations between tensor spaces
│   └── spaces.py      # Pre-defined tensor spaces
│
├── minisimulator/     # Minimal simulation utilities
│
├── cafemodel/        # Example cafe sustainability model
│
└── trafficmodel/     # Example traffic demand model
```

## Type Safety Guarantees

The type system enforces:

1. **Tensor space compatibility**:
   - Operations only allowed between tensors of the same space
   - Compile-time detection of space mismatches

2. **Morphism correctness**:
   - Source and destination spaces must be compatible
   - Transformation axes must match the spaces' dimensions

3. **Basis consistency**:
   - Each basis defines its axes as a tuple of integers
   - Ensures consistent dimensional semantics

Example from the code:
```python
def add(self, t1: Tensor[B], t2: Tensor[B]) -> Tensor[B]:
    """Element-wise addition of two tensors."""
    ensure_same_basis(self.basis, t1.space.basis)
    ensure_same_basis(self.basis, t2.space.basis)
    return self.new_tensor(graph.add(t1.node, t2.node))
```

## Example Models

The package includes two example models demonstrating the framework:

1. **Cafe Model** (`cafemodel`):
   - Models cafe operations with capacity constraints
   - Demonstrates enumerations for weather and time of day
   - Shows sustainability analysis across multiple contexts

2. **Traffic Model** (`trafficmodel`):
   - Models traffic demand patterns with price sensitivity
   - Demonstrates time-shifting effects across multiple dimensions
   - Shows complex tensor operations and projections

## Migration Path

To integrate into dt-model:

1. Introduce type-safe tensor spaces to critical components first

## Future Work

1. TensorFlow/PyTorch backend implementation
2. Just-in-time compilation for performance optimization
3. Distributed computation support for large models
4. Extended visualization and analysis tools

## Conclusion

YAKOF demonstrates a path forward for sustainability modeling that:
- Makes dimension errors impossible through the type system
- Provides clear semantics for tensor transformations
- Supports multiple backends and evaluation strategies

By focusing on tensor space semantics rather than just shapes,
the framework enables more robust and maintainable sustainability models.
