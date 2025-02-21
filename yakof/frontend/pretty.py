"""
Pretty Printing for Computation Graphs
======================================

This module provides facilities for converting computation graphs into
readable string representations. It handles:

1. Operator precedence
2. Parentheses insertion
3. Special formatting for function-like operations
4. Named expressions

The main entry point is the format() function:

    >>> from yakof.frontend import graph, pretty
    >>> x = graph.placeholder("x")
    >>> y = graph.add(graph.multiply(x, 2), 1)
    >>> print(pretty.format(y))
    x * 2 + 1

Precedence Rules
---------------

The formatter follows standard mathematical precedence:

1. Function application (exp, log)
2. Unary operations (~)
3. Power (**)
4. Multiply/Divide (*, /)
5. Add/Subtract (+, -)
6. Comparisons (<, <=, >, >=, ==, !=)
7. Logical AND (&)
8. Logical OR/XOR (|, ^)

Parentheses are automatically added when needed to preserve
the correct evaluation order:

    >>> x + y * z      # "x + y * z"
    >>> (x + y) * z    # "(x + y) * z"
    >>> ~x & y | z     # "(~x & y) | z"

Design Decisions
---------------
1. Precedence-based Formatting:
   - Uses numeric precedence levels to determine parenthesization
   - Follows standard mathematical conventions
   - Allows easy addition of new operators

2. Recursive Implementation:
   - Handles nested expressions naturally
   - Passes precedence information down the tree
   - Enables context-aware formatting decisions

3. Special Cases:
   - Function-like operations use function call syntax
   - Named nodes show assignment syntax
   - Placeholders use angle bracket notation for visibility

Implementation Notes
------------------
The formatter uses a visitor-like pattern without explicitly implementing
the visitor pattern, which keeps the code simpler while maintaining
extensibility.
"""

# SPDX-License-Identifier: Apache-2.0

from yakof.frontend import graph


def format(node: graph.Node) -> str:
    """Format a computation graph node as a string.

    Args:
        node: The node to format

    Returns:
        A string representation with appropriate parentheses
        and operator precedence.

    Examples:
        >>> x = graph.placeholder("x")
        >>> y = graph.add(graph.multiply(x, 2), 1)
        >>> print(pretty.format(y))
        x * 2 + 1
    """
    expr = _format(node, 0)  # Start with lowest precedence
    if node.name:
        expr = f"{node.name} = {expr}"
    return expr


def _format(node: graph.Node, parent_precedence: int) -> str:
    """Internal recursive formatter.

    Args:
        node: The node to format
        parent_precedence: The precedence of the parent operation

    Returns:
        Formatted string with appropriate parentheses based on
        operator precedence.
    """
    # Precedence rules (higher binds tighter)
    PRECEDENCE = {
        # Unary operations
        graph.logical_not: 50,  # ~x
        graph.exp: 50,  # exp(x)
        graph.log: 50,  # log(x)
        # Binary operations
        graph.power: 40,  # x ** y
        graph.multiply: 30,  # x * y
        graph.divide: 30,  # x / y
        graph.add: 20,  # x + y
        graph.subtract: 20,  # x - y
        # Comparisons
        graph.less: 10,  # x < y
        graph.less_equal: 10,  # x <= y
        graph.greater: 10,  # x > y
        graph.greater_equal: 10,  # x >= y
        graph.equal: 10,  # x == y
        graph.not_equal: 10,  # x != y
        # Logical operations
        graph.logical_and: 5,  # x & y
        graph.logical_or: 4,  # x | y
        graph.logical_xor: 4,  # x ^ y
    }

    def needs_parens(node: graph.Node) -> bool:
        """Determine if expression needs parentheses."""
        return PRECEDENCE.get(type(node), 0) < parent_precedence

    def wrap(expr: str) -> str:
        """Wrap expression in parentheses if needed."""
        return f"({expr})" if needs_parens(node) else expr

    # Base cases
    if isinstance(node, graph.constant):
        if node.name:
            return node.name
        return str(node.value)
    if isinstance(node, graph.placeholder):
        return node.name

    # Binary operations
    if isinstance(node, graph.BinaryOp):
        op_precedence = PRECEDENCE.get(type(node), 0)
        left = _format(node.left, op_precedence)
        right = _format(node.right, op_precedence)

        # Arithmetic operators
        if isinstance(node, graph.add):
            return wrap(f"{left} + {right}")
        if isinstance(node, graph.subtract):
            return wrap(f"{left} - {right}")
        if isinstance(node, graph.multiply):
            return wrap(f"{left} * {right}")
        if isinstance(node, graph.divide):
            return wrap(f"{left} / {right}")
        if isinstance(node, graph.power):
            return wrap(f"{left} ** {right}")
        if isinstance(node, graph.logical_and):
            return wrap(f"{left} & {right}")
        if isinstance(node, graph.logical_or):
            return wrap(f"{left} | {right}")
        if isinstance(node, graph.logical_xor):
            return wrap(f"{left} ^ {right}")

        # Comparison operators
        if isinstance(node, graph.less):
            return wrap(f"{left} < {right}")
        if isinstance(node, graph.less_equal):
            return wrap(f"{left} <= {right}")
        if isinstance(node, graph.greater):
            return wrap(f"{left} > {right}")
        if isinstance(node, graph.greater_equal):
            return wrap(f"{left} >= {right}")
        if isinstance(node, graph.equal):
            return wrap(f"{left} == {right}")
        if isinstance(node, graph.not_equal):
            return wrap(f"{left} != {right}")

    # Unary operations
    if isinstance(node, graph.UnaryOp):
        op_precedence = PRECEDENCE.get(type(node), 0)
        inner = _format(node.node, op_precedence)

        if isinstance(node, graph.logical_not):
            return wrap(f"~{inner}")
        if isinstance(node, graph.exp):
            return f"exp({inner})"
        if isinstance(node, graph.log):
            return f"log({inner})"

    # Function-like operations
    if isinstance(node, graph.normal_cdf):
        return f"normal_cdf({_format(node.x, 0)}, loc={_format(node.loc, 0)}, scale={_format(node.scale, 0)})"
    if isinstance(node, graph.uniform_cdf):
        return f"uniform_cdf({_format(node.x, 0)}, loc={_format(node.loc, 0)}, scale={_format(node.scale, 0)})"

    return f"<unknown:{type(node).__name__}>"
