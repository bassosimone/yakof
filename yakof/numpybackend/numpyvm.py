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

from . import numpylower


Bindings = dict[str, np.ndarray]
"""Type alias for a dictionary of variable bindings."""

RegisterFile = dict[numpylower.Register, np.ndarray]
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
        program: numpylower.Program,
        target_register: numpylower.Register,
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

    def _execute_operation(self, op: numpylower.Operation, bindings: Bindings) -> None:
        """Execute single operation and store result in appropriate register."""

        if isinstance(op, numpylower.constant):
            self.registers[len(self.registers)] = op.value
            return

        if isinstance(op, numpylower.placeholder):
            if op.name not in bindings:
                if op.default_value is not None:
                    self.registers[len(self.registers)] = op.default_value
                    return
                raise ValueError(
                    f"numpyvm: no value provided for placeholder '{op.name}'"
                )
            self.registers[len(self.registers)] = bindings[op.name]
            return

        # Binary operations
        if isinstance(op, numpylower.BinaryOp):
            left = self.registers[op.left]
            right = self.registers[op.right]

            ops = {
                numpylower.add: np.add,
                numpylower.subtract: np.subtract,
                numpylower.multiply: np.multiply,
                numpylower.divide: np.divide,
                numpylower.equal: np.equal,
                numpylower.not_equal: np.not_equal,
                numpylower.less: np.less,
                numpylower.less_equal: np.less_equal,
                numpylower.greater: np.greater,
                numpylower.greater_equal: np.greater_equal,
                numpylower.logical_and: np.logical_and,
                numpylower.logical_or: np.logical_or,
                numpylower.logical_xor: np.logical_xor,
                numpylower.power: np.power,
                numpylower.maximum: np.maximum,
            }

            try:
                self.registers[len(self.registers)] = ops[type(op)](left, right)
            except KeyError:
                raise TypeError(f"numpyvm: unknown binary operation: {type(op)}")

        # Unary operations
        if isinstance(op, numpylower.UnaryOp):
            operand = self.registers[op.register]

            ops = {
                numpylower.logical_not: np.logical_not,
                numpylower.exp: np.exp,
                numpylower.log: np.log,
            }

            try:
                self.registers[len(self.registers)] = ops[type(op)](operand)
            except KeyError:
                raise TypeError(f"numpyvm: unknown unary operation: {type(op)}")

        # Conditional operations
        if isinstance(op, numpylower.where):
            self.registers[len(self.registers)] = np.where(
                self.registers[op.condition],
                self.registers[op.then],
                self.registers[op.otherwise],
            )

        if isinstance(op, numpylower.multi_clause_where):
            conditions = []
            values = []
            for cond_reg, value_reg in op.clauses[:-1]:
                conditions.append(self.registers[cond_reg])
                values.append(self.registers[value_reg])
            default = self.registers[op.clauses[-1][1]]
            self.registers[len(self.registers)] = np.select(
                conditions, values, default=default
            )

        # Shape operations
        if isinstance(op, numpylower.reshape):
            self.registers[len(self.registers)] = self.registers[
                op.register
            ].reshape(op.shape)

        # Axis operations
        if isinstance(op, numpylower.AxisOp):
            operand = self.registers[op.register]

            ops = {
                numpylower.expand_dims: lambda x: np.expand_dims(x, op.axis),
                numpylower.reduce_sum: lambda x: np.sum(x, axis=op.axis),
                numpylower.reduce_mean: lambda x: np.mean(x, axis=op.axis),
            }

            try:
                self.registers[len(self.registers)] = ops[type(op)](operand)
            except KeyError:
                raise TypeError(f"numpyvm: unknown axis operation: {type(op)}")
