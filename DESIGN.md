# YAKOF (Yet Another Kettle of Fish)

## Overview

YAKOF is a technology demonstrator showing how we could evolve the existing
dt-model package. The key improvements demonstrated are:

1. Replacement of SymPy with a focused AST framework
2. Type-safe parameter space modeling
3. Orientation-aware tensor operations

## Design Philosophy

This project follows these design principles:

- Favor simple, composable components over complex, monolithic solutions
- Handle the common case well and let exceptional cases fail naturally
- Put complexity in the type system, not in the runtime code

## Core Design Decisions

### 1. Custom AST Framework

**Problem**: The original implementation used SymPy for symbolic computation, which:

- Created an artificial boundary between symbolic and numeric operations
- Required complex shape manipulation to bridge domains
- Made it difficult to naturally express NumPy/TensorFlow operations

**Solution**: A minimal AST framework that:

- Maps directly to backend (NumPy/TensorFlow) operations
- Handles tensors and shapes naturally
- Can support multiple backends (currently NumPy, potentially TensorFlow)

The AST framework is intentionally limited to operations needed for sustainability
modeling, avoiding the complexity of a general symbolic computation system.

### 2. Parameter Space Modeling

**Problem**: The original implementation required careful management of:
- Dependency ordering between equations
- Explicit shape manipulation
- Type safety across different kinds of parameters

**Solution**: Type-safe parameter space modeling that:
- Uses numpy.meshgrid for natural parameter space representation
- Eliminates explicit dependency ordering
- Provides compile-time type safety

### 3. Tensor Orientations

**Problem**: In sustainability modeling, tensors can represent:
- Time series of input data
- Ensemble variations
- Field combinations of both

Mixing these incorrectly leads to subtle bugs. Additionally, one needs
to carefully craft evaluation and correctly lift/project tensors.

**Solution**: Orientation-aware tensor system that:
- Encodes tensor orientation directly in the type system
- Makes invalid combinations a compile-time error
- Provides explicit lifting/projecting operations between spaces

## Architecture

The architecture is intentionally "wide" rather than "tall":

```
backend/
  graph.py          # Core tensor operations
  orientation.py    # Typed tensor orientations
  numpy_engine.py   # NumPy evaluation backend

# High-level modeling APIs focused on existing use cases
fieldspace/
phasespace/
```

Adding new operations requires only:
1. Adding operation to graph.py
2. Implementing evaluation in numpy_engine.py
3. (Optional) Adding to orientation.py if needed

## Type Safety Guarantees

The type system enforces:
1. Tensor orientation safety
2. Parameter space consistency
3. Backend operation compatibility

Runtime checks are minimal and focused on:
1. Input validation where required by the domain
2. Letting underlying numerical operations fail naturally

## Testing Strategy

Following these principles:
1. Test the core tensor operations thoroughly
2. Test type safety at compile time
3. Let numerical errors surface through NumPy
4. Focus integration tests on modeling patterns

## Migration Path

To integrate into dt-model
1. Document design decisions, discuss and reach consensus
2. Create specific issues for integration
3. Implement and merging changes incrementally
4. Maintain backwards compatibility unless impractical

## Future Work

1. TensorFlow backend implementation
2. Additional numerical operation support as needed
3. Documentation of modeling patterns
4. Parallel operations using Dask or similar

## Conclusion

YAKOF demonstrates how focused improvements in type safety and architectural clarity
can evolve dt-model while maintaining simplicity and composability. The design choices
prioritize:

1. Making invalid states unrepresentable and reducing the amount of code
required to reshape tensors according to the problem's geometry
2. Keeping the implementation simple and maintainable
3. Supporting natural expression of sustainability models
