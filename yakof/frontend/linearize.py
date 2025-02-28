"""
Linearization of Computation Graphs
===================================

This module provides functions to linearize computation graphs into execution plans.
It performs topological sorting of graph nodes, ensuring dependencies are evaluated
before the nodes that depend on them.

The linearization process:
1. Starts from output nodes and traverses the graph
2. Ensures all dependencies are scheduled before their dependents
3. Maintains creation order where possible (for nodes with no dependency relationship)
4. Handles common graph structures (binary operations, conditionals, etc.)

This is useful for:
- Creating efficient execution plans for evaluators
- Visualizing the computation flow in order
- Debugging models by inspecting operations in a logical sequence
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from . import graph


def forest(*leaves: graph.Node) -> list[graph.Node]:
    """
    Linearize a computation forest (multiple output nodes) into an execution plan.

    In a computation graph, "leaves" refer to the output nodes that have no
    dependents (nothing depends on them). These are typically the final results
    of your computation. We start linearization from these leaf nodes and work
    backwards to find all dependencies.

    This function creates a linearized execution plan from multiple leaf/output nodes,
    ensuring all dependencies are evaluated before the nodes that depend on them.
    When multiple paths exist through the graph, the original creation order is
    preserved where possible.

    Args:
        *leaves: Output/leaf nodes of the computation graph as separate arguments.
                These should be the final outputs of your computation.
                Use unpacking (*list_of_nodes) to pass a list of nodes.

    Returns:
        Topologically sorted list of nodes forming an execution plan

    Raises:
        ValueError: If a cycle is detected in the graph
        TypeError: If an unknown node type is encountered

    Examples:
        >>> # Single output
        >>> plan = linearize.forest(output_node)
        >>>
        >>> # Multiple outputs
        >>> plan = linearize.forest(output1, output2, output3)
        >>>
        >>> # List of outputs
        >>> plan = linearize.forest(*output_list)
    """
    plan: list[graph.Node] = []
    visiting: set[graph.Node] = set()  # For cycle detection
    visited: set[graph.Node] = set()

    def visit(node: graph.Node) -> None:
        if node in visited:
            return

        # Check for cycles
        if node in visiting:
            raise ValueError(
                f"linearize: cycle detected in computation graph at node {node.name or f'<unnamed node {node.id}>'}"
            )

        visiting.add(node)

        # Get dependencies based on node type
        deps = _get_dependencies(node)

        # Sort dependencies by creation ID for deterministic ordering
        # when multiple paths exist
        deps.sort(key=lambda n: n.id)

        # Visit all dependencies first
        for dep in deps:
            visit(dep)

        # Done with this node
        visiting.remove(node)
        visited.add(node)
        plan.append(node)

    # Sort inputs by creation ID for deterministic ordering
    for node in sorted(leaves, key=lambda n: n.id):
        visit(node)

    # Sort outputs by creation ID for deterministic ordering
    return sorted(plan, key=lambda n: n.id)


def _get_dependencies(node: graph.Node) -> list[graph.Node]:
    """
    Get the direct dependencies of a node.

    Args:
        node: The node to get dependencies for

    Returns:
        List of nodes that are direct dependencies

    Raises:
        TypeError: If the node type is unknown
    """
    if isinstance(node, graph.BinaryOp):
        return [node.left, node.right]
    elif isinstance(node, graph.UnaryOp):
        return [node.node]
    elif isinstance(node, graph.where):
        return [node.condition, node.then, node.otherwise]
    elif isinstance(node, graph.multi_clause_where):
        deps = [node.default_value]
        for cond, value in node.clauses:
            deps.append(cond)
            deps.append(value)
        return deps
    elif isinstance(node, graph.AxisOp):
        return [node.node]
    elif isinstance(node, (graph.constant, graph.placeholder)):
        return []
    else:
        raise TypeError(f"linearize: unknown node type: {type(node)}")
