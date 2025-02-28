"""
NumPy Virtual Machine
=====================

The virtual machine approach provides several benefits:

1. Caching:
   - Intermediate results stored in registers
   - Results reusable across multiple evaluations
   - Partial evaluation support
   - Memory efficiency through register reuse

2. Execution Control:
   - Fine-grained control over evaluation order
   - Lazy evaluation of only needed results
   - Foundation for future optimizations

3. State Management:
   - Clear lifecycle for intermediate results
   - Explicit memory management
   - Protected internal state

Design Decisions
----------------

1. Register-based:
   - Easier to optimize than stack-based
   - Natural fit for tensor operations
   - Clear data dependencies

2. Stateful execution:
   - Enables caching across evaluations
   - Clear separation of compilation and execution
   - Explicit control over resource usage

3. Lazy evaluation:
   - Only compute needed results
   - Support for partial evaluation
   - Foundation for future optimizations
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from . import emitter


Bindings = dict[str, np.ndarray]
"""Type alias for a dictionary of variable bindings."""

RegisterFile = dict[emitter.Register, np.ndarray]
"""Type alias for register storage."""


class VirtualMachine:
    """Virtual machine for executing linearized NumPy programs."""

    def __init__(self):
        self.registers: RegisterFile = {}

    def clear_registers(self) -> None:
        """Clear all registers."""
        self.registers.clear()

    def execute(
        self,
        program: emitter.Program,
        target_register: emitter.Register,
        bindings: Bindings,
    ) -> np.ndarray:
        """
        Execute program up to target register.

        Args:
            program: Linearized program to execute
            target_register: Register containing the final result
            bindings: Variable bindings for placeholders

        Returns:
            Content of the target register after execution
        """
        # Execute only operations needed for target register
        for idx, op in enumerate(program.operations[: target_register + 1]):
            if idx in self.registers:
                continue  # Skip if already cached

            self._execute_operation(op, bindings)

        return self.registers[target_register]

    def _execute_operation(self, op: emitter.Operation, bindings: Bindings) -> None:
        """Execute single operation and store result in appropriate register."""

        if isinstance(op, emitter.constant):
            self.registers[len(self.registers)] = op.value
            return

        if isinstance(op, emitter.placeholder):
            if op.name not in bindings:
                if op.default_value is not None:
                    self.registers[len(self.registers)] = op.default_value
                    return
                raise ValueError(f"vm: no value provided for placeholder '{op.name}'")
            self.registers[len(self.registers)] = bindings[op.name]
            return

        # Binary operations
        if isinstance(op, emitter.BinaryOp):
            left = self.registers[op.left]
            right = self.registers[op.right]

            ops = {
                emitter.add: np.add,
                emitter.subtract: np.subtract,
                emitter.multiply: np.multiply,
                emitter.divide: np.divide,
                emitter.equal: np.equal,
                emitter.not_equal: np.not_equal,
                emitter.less: np.less,
                emitter.less_equal: np.less_equal,
                emitter.greater: np.greater,
                emitter.greater_equal: np.greater_equal,
                emitter.logical_and: np.logical_and,
                emitter.logical_or: np.logical_or,
                emitter.logical_xor: np.logical_xor,
                emitter.power: np.power,
                emitter.maximum: np.maximum,
            }

            try:
                self.registers[len(self.registers)] = ops[type(op)](left, right)
            except KeyError:
                raise TypeError(f"vm: unknown binary operation: {type(op)}")

        # Unary operations
        if isinstance(op, emitter.UnaryOp):
            operand = self.registers[op.register]

            ops = {
                emitter.logical_not: np.logical_not,
                emitter.exp: np.exp,
                emitter.log: np.log,
            }

            try:
                self.registers[len(self.registers)] = ops[type(op)](operand)
            except KeyError:
                raise TypeError(f"vm: unknown unary operation: {type(op)}")

        # Conditional operations
        if isinstance(op, emitter.where):
            self.registers[len(self.registers)] = np.where(
                self.registers[op.condition],
                self.registers[op.then],
                self.registers[op.otherwise],
            )

        if isinstance(op, emitter.multi_clause_where):
            conditions = []
            values = []
            for cond_reg, value_reg in op.clauses[:-1]:
                conditions.append(self.registers[cond_reg])
                values.append(self.registers[value_reg])
            default = self.registers[op.clauses[-1][1]]
            self.registers[len(self.registers)] = np.select(
                conditions, values, default=default
            )

        # Axis operations
        if isinstance(op, emitter.AxisOp):
            operand = self.registers[op.register]

            ops = {
                emitter.expand_dims: lambda x: np.expand_dims(x, op.axis),
                emitter.reduce_sum: lambda x: np.sum(x, axis=op.axis),
                emitter.reduce_mean: lambda x: np.mean(x, axis=op.axis),
            }

            try:
                self.registers[len(self.registers)] = ops[type(op)](operand)
            except KeyError:
                raise TypeError(f"vm: unknown axis operation: {type(op)}")
