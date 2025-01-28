"""
Pretty formatting utilities for tensor graphs.

SPDX-License-Identifier: Apache-2.0
"""

from . import graph


def format(tensor: graph.Tensor) -> str:
    """Format a tensor expression as a readable string.

    Args:
        tensor: The tensor expression to format.

    Returns:
        A readable string representation of the tensor expression,
        with named subexpressions preserved for clarity.

    Example:
        >>> x = graph.placeholder("x")
        >>> y = x * 2 + 1
        >>> print(format(y))
        y = x * 2 + 1
    """
    if isinstance(tensor, graph.placeholder):
        return f"{_format(tensor, True)}"
    return f"{tensor.name} = {_format(tensor, True)}"


def _format(node: graph.Tensor, toplev: bool) -> str:
    """Internal recursive formatter.

    Args:
        node: The node to format
        toplev: Whether this is a top-level expression

    Returns:
        Formatted string for the node and its children.
    """
    # Use existing names for subexpressions
    if node.name and not toplev:
        return node.name

    # Handle different node types
    if isinstance(node, graph.placeholder):
        if hasattr(node, "default_value") and node.default_value is not None:
            return f"{node.name} = {_format(node.default_value, False)}"
        return f"{node.name} = <placeholder>"

    if isinstance(node, graph.constant):
        return str(node.get_value())

    # Binary arithmetic operations
    if isinstance(node, graph.add):
        return f"{_format(node.left, False)} + {_format(node.right, False)}"

    if isinstance(node, graph.subtract):
        return f"{_format(node.left, False)} - {_format(node.right, False)}"

    if isinstance(node, graph.multiply):
        return f"{_format(node.left, False)} * {_format(node.right, False)}"

    if isinstance(node, graph.divide):
        return f"{_format(node.left, False)} / {_format(node.right, False)}"

    # Comparison operations
    if isinstance(node, graph.equal):
        return f"{_format(node.left, False)} == {_format(node.right, False)}"

    if isinstance(node, graph.not_equal):
        return f"{_format(node.left, False)} != {_format(node.right, False)}"

    if isinstance(node, graph.less):
        return f"{_format(node.left, False)} < {_format(node.right, False)}"

    if isinstance(node, graph.less_equal):
        return f"{_format(node.left, False)} <= {_format(node.right, False)}"

    if isinstance(node, graph.greater):
        return f"{_format(node.left, False)} > {_format(node.right, False)}"

    if isinstance(node, graph.greater_equal):
        return f"{_format(node.left, False)} >= {_format(node.right, False)}"

    # Logical operations
    if isinstance(node, graph.logical_and):
        return f"{_format(node.left, False)} & {_format(node.right, False)}"

    if isinstance(node, graph.logical_or):
        return f"{_format(node.left, False)} | {_format(node.right, False)}"

    if isinstance(node, graph.logical_xor):
        return f"{_format(node.left, False)} ^ {_format(node.right, False)}"

    if isinstance(node, graph.logical_not):
        return f"~{_format(node.x, False)}"

    # Array operations
    if isinstance(node, graph.reshape):
        return f"reshape({_format(node.x, False)}, shape={node.new_shape})"

    if isinstance(node, graph.expand_dims):
        return f"expand_dims({_format(node.x, False)}, axis={node.axis})"

    if isinstance(node, graph.squeeze):
        return f"squeeze({_format(node.x, False)}, axis={node.axes})"

    if isinstance(node, graph.reduce_sum):
        return f"sum({_format(node.x, False)}, axis={node.axes})"

    # Control flow
    if isinstance(node, graph.where):
        return (
            f"where({_format(node.condition, False)}, "
            f"{_format(node.x, False)}, {_format(node.y, False)})"
        )

    # Random operations
    if isinstance(node, graph.uniform_rvs):
        return (
            f"random_uniform(shape={node.shape}, "
            f"loc={_format(node.loc, False)}, "
            f"scale={_format(node.scale, False)})"
        )

    if isinstance(node, graph.uniform_cdf):
        return (
            f"cdf_random({_format(node.x, False)}, "
            f"loc={_format(node.loc, False)}, "
            f"scale={_format(node.scale, False)})"
        )

    if isinstance(node, graph.normal_rvs):
        return f"normal_rvs(shape={node.shape}, loc={_format(node.loc, False)}, scale={_format(node.scale, False)})"

    if isinstance(node, graph.normal_cdf):
        return f"normal_cdf({_format(node.x, False)}, loc={_format(node.loc, False)}, scale={_format(node.scale, False)})"

    if isinstance(node, graph.maximum):
        return f"maximum({_format(node.x, False)}, {_format(node.y, False)})"

    if isinstance(node, graph.reduce_mean):
        return f"mean({_format(node.x, False)}, axis={node.axis})"

    if isinstance(node, graph.multi_clause_where):
        cases = [f"({_format(c, False)}, {_format(v, False)})" for c, v in node.cases]
        return f"multi_clause_where({', '.join(cases)})"

    # TODO(bassosimone): add more node types here if we see
    # that pretty printing them leads to `unknown`.

    return f"unknown: {node.name}"
