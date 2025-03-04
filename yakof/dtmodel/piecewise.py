"""
Piecewise Implementation
========================

This module provides the basic building block for emulating the sympy
piecewise function using the tensor language frontend.
"""

from ..frontend import graph

from . import geometry


Cond = geometry.ComputationTensor | graph.Scalar
"""Condition for a piecewise clause."""

Expr = geometry.ComputationTensor | graph.Scalar
"""Expression for a piecewise clause."""

Clause = tuple[Expr, Cond]
"""Clause provided to piecewise."""


def to_tensor(*clauses: Clause) -> geometry.ComputationTensor:
    """Converts the provide clauses arranged according to the sympy.Piecewise
    convention into a graph.multi_clause_where computation tensor.

    Args:
        *clauses: The clauses to be converted.

    Returns:
        The computation tensor representing the piecewise function.

    Raises:
        ValueError: If no clauses are provided.
    """
    # Ensure that we remove all the clauses after a true clause
    return _to_tensor(_filter_clauses(clauses))


def _filter_clauses(clauses: tuple[Clause, ...]) -> list[Clause]:
    """This function removes the clauses after the first true clause, to
    correctly emulate the sumpy.Piecewise behaviour.

    Args:
        clauses: The clauses to be filtered.

    Returns:
        The filtered clauses.
    """
    filtered: list[Clause] = []
    for expr, cond in clauses:
        filtered.append((expr, cond))
        if cond is True:
            break
    return filtered


def _to_tensor(clauses: list[Clause]) -> geometry.ComputationTensor:
    # 1. Bail if there are no remaining clauses
    if len(clauses) < 1:
        raise ValueError("piecewise: at least one clause is required")

    # 2. Check whether there is a default case and otherwise use NaN
    default_value: Expr = float("NaN")
    last_clause = clauses[-1]
    if last_clause[1] is True:
        default_value = last_clause[0]
        clauses = clauses[:-1]
    if isinstance(default_value, graph.Scalar):
        default_value = geometry.ComputationSpace.constant(default_value)

    # 3. Prepare the reversed clauses adapting the types
    reversed: list[tuple[geometry.ComputationTensor, geometry.ComputationTensor]] = []
    for expr, cond in clauses:
        if isinstance(expr, graph.Scalar):
            expr = geometry.ComputationSpace.constant(expr)
        if isinstance(cond, graph.Scalar):
            cond = geometry.ComputationSpace.constant(cond)
        reversed.append((cond, expr))

    # 4. We're now all set call multi_clause_where
    return geometry.ComputationSpace.multi_clause_where(reversed, default_value)
