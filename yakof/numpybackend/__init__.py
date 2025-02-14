"""
NumPy Backend
=============

Interpreter
-----------

This backend evaluates tensor computations by calling NumPy
functions through the following stages:

1. Translation to NumPy Semantic tree (semtree.py)
   - Provides NumPy-specific operation types
   - Enables backend-specific optimizations

2. Interpretation (evaluator.py)

The NumPy SemTree stage is shared between the interpreter and the compiler.

Compiler
--------

This backend translates tensor computations into NumPy operations
through a three-stage compilation process:

1. Translation to NumPy Semantic Tree (semtree.py)
   - Provides NumPy-specific operation types
   - Enables backend-specific optimizations
   - Acts as bridge between frontend and linear form

2. Lowering to linear form (emitter.py)
   - Flattens nested expressions into sequence of operations
   - Introduces explicit virtual registers
   - Enables caching of intermediate results
   - Makes data dependencies explicit

3. Virtual machine execution (vm.py)
   - Caches intermediate results in registers
   - Enables partial evaluation of computation graph
   - Allows reuse of computed values across evaluations
   - Provides foundation for future optimizations

Design Philosophy:
----------------
The multi-stage compilation allows separating concerns:
- Frontend can focus on type safety and ergonomics
- SemTree potentially enables backend-specific optimizations
- Linear form makes data flow explicit
- VM provides execution efficiency
"""

# SPDX-License-Identifier: Apache-2.0
