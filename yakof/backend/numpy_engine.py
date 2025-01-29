"""
NumPy tensor graph evaluation engine
====================================

This module implements a NumPy tensor graph evaluation engine. The input to
the evaluation is a graph.Tensor along with a set of bindings for placeholders,
while the output is a numpy.ndarray. The evaluation is potentially memoized
using an optional caching functionality (disabled by default).

This code also implements debugging, which allows one to evaluate each tensor
in a graph independently, printing its name, shape, and value. This functionality
helps in debugging and understanding what the underlying code is doing.

SPDX-License-Identifier: Apache-2.0
"""

from typing import Dict, Iterator, Mapping, Protocol, runtime_checkable
from dataclasses import dataclass
from scipy import stats

import numpy as np

from . import graph

Array = np.ndarray
ScalarOrArray = float | int | bool | Array
Bindings = Mapping[str, ScalarOrArray]

# --- Cache Protocol and Implementations ---


class Cache(Protocol):
    """Protocol for caching evaluation results."""

    def get(self, key: graph.Tensor) -> Array | None: ...

    def put(self, key: graph.Tensor, value: Array) -> None: ...


class DictCache:
    """Simple dictionary-based cache implementation.

    Note: every expression is evaluated at most once and cached
    including conditional expressions. This leads to perfectly
    reproducible cache-reified execution traces. To evaluate under
    different conditions, use different cache instances."""

    def __init__(self):
        self._cache: Dict[graph.Tensor, Array] = {}

    def get(self, key: graph.Tensor) -> Array | None:
        return self._cache.get(key)

    def put(self, key: graph.Tensor, value: Array) -> None:
        self._cache[key] = value


class NullCache:
    """Non-caching implementation for single-shot evaluation."""

    def get(self, key: graph.Tensor) -> Array | None:
        return None

    def put(self, key: graph.Tensor, value: Array) -> None:
        pass


# --- Core Evaluation Logic ---


def _get_operands(node: graph.Tensor) -> dict[str, graph.Tensor]:
    """Get named operands for a node (for debugging)."""
    if isinstance(node, (graph.constant, graph.placeholder)):
        return {}

    if isinstance(
        node,
        (
            graph.add,
            graph.subtract,
            graph.multiply,
            graph.divide,
            graph.equal,
            graph.not_equal,
            graph.less,
            graph.less_equal,
            graph.greater,
            graph.greater_equal,
            graph.logical_and,
            graph.logical_or,
            graph.logical_xor,
        ),
    ):
        return {"left": node.left, "right": node.right}

    if isinstance(
        node,
        (
            graph.logical_not,
            graph.reshape,
            graph.expand_dims,
            graph.squeeze,
            graph.reduce_sum,
        ),
    ):
        return {"x": node.x}

    if isinstance(node, graph.where):
        return {"condition": node.condition, "x": node.x, "y": node.y}

    if isinstance(node, graph.uniform_rvs):
        return {"loc": node.loc, "scale": node.scale}

    if isinstance(node, graph.uniform_cdf):
        return {"x": node.x, "loc": node.loc, "scale": node.scale}

    return {}


def _evaluate(
    node: graph.Tensor, bindings: Bindings, cache: Cache, debug: bool = False
) -> Array:
    """Core evaluation function - pure and explicit."""

    # Check cache first
    if (cached := cache.get(node)) is not None:
        if debug and node.name:
            print(f"\n--- begin cached eval trace ---")
            print(f"→ Formula: {node.name}")
            print(f"→ Using cached value")
            print(f"→ Result:\n{cached}")
            print(f"--- end cached eval trace ---")
        return cached

    # Pre-evaluation debug info
    if debug and node.name:
        operands = _get_operands(node)
        print(f"\n--- begin eval trace ---")
        print(f"→ Formula: {node.name}")
        print(f"→ Operator: {type(node).__name__}")
        if operands:
            shapes = {
                name: _evaluate(op, bindings, cache).shape
                for name, op in operands.items()
            }
            print(f"→ Operand shapes: {shapes}")
        print(f"→ Beginning evaluation...")

    # Evaluate node
    try:
        if isinstance(node, graph.constant):
            result = np.array(node.get_value())

        elif isinstance(node, graph.placeholder):
            if node.name not in bindings:
                if node.default_value is not None:
                    result = _evaluate(node.default_value, bindings, cache)
                else:
                    raise ValueError(f"No value provided for placeholder '{node.name}'")
            else:
                result = np.asarray(bindings[node.name])

        elif isinstance(node, graph.add):
            result = _evaluate(node.left, bindings, cache) + _evaluate(
                node.right, bindings, cache
            )

        elif isinstance(node, graph.subtract):
            result = _evaluate(node.left, bindings, cache) - _evaluate(
                node.right, bindings, cache
            )

        elif isinstance(node, graph.multiply):
            result = _evaluate(node.left, bindings, cache) * _evaluate(
                node.right, bindings, cache
            )

        elif isinstance(node, graph.divide):
            result = _evaluate(node.left, bindings, cache) / _evaluate(
                node.right, bindings, cache
            )

        elif isinstance(
            node,
            (
                graph.equal,
                graph.not_equal,
                graph.less,
                graph.less_equal,
                graph.greater,
                graph.greater_equal,
            ),
        ):
            left = _evaluate(node.left, bindings, cache)
            right = _evaluate(node.right, bindings, cache)
            if isinstance(node, graph.equal):
                result = left == right
            elif isinstance(node, graph.not_equal):
                result = left != right
            elif isinstance(node, graph.less):
                result = left < right
            elif isinstance(node, graph.less_equal):
                result = left <= right
            elif isinstance(node, graph.greater):
                result = left > right
            else:  # greater_equal
                result = left >= right

        elif isinstance(node, (graph.logical_and, graph.logical_or, graph.logical_xor)):
            left = _evaluate(node.left, bindings, cache)
            right = _evaluate(node.right, bindings, cache)
            if isinstance(node, graph.logical_and):
                result = left & right
            elif isinstance(node, graph.logical_or):
                result = left | right
            else:  # logical_xor
                result = left ^ right

        elif isinstance(node, graph.logical_not):
            result = ~_evaluate(node.x, bindings, cache)

        elif isinstance(node, graph.reshape):
            result = _evaluate(node.x, bindings, cache).reshape(node.new_shape)

        elif isinstance(node, graph.expand_dims):
            result = np.expand_dims(_evaluate(node.x, bindings, cache), node.axis)

        elif isinstance(node, graph.squeeze):
            result = np.squeeze(_evaluate(node.x, bindings, cache), node.axes)

        elif isinstance(node, graph.reduce_sum):
            result = _evaluate(node.x, bindings, cache).sum(axis=node.axes)

        elif isinstance(node, graph.where):
            result = np.where(
                _evaluate(node.condition, bindings, cache),
                _evaluate(node.x, bindings, cache),
                _evaluate(node.y, bindings, cache),
            )

        elif isinstance(node, (graph.cond, graph.multi_clause_where)):
            # Implementation note: np.select evaluates in order so it maps
            # quite nicely to the scheme-like semantics of `cond`.
            conditions = []
            values = []
            for cond, value in node.cases[:-1]:
                conditions.append(_evaluate(cond, bindings, cache))
                values.append(_evaluate(value, bindings, cache))
            default = _evaluate(node.cases[-1][1], bindings, cache)
            result = np.select(conditions, values, default=default)

        elif isinstance(node, graph.uniform_rvs):
            result = np.array(
                stats.uniform.rvs(
                    loc=_evaluate(node.loc, bindings, cache),
                    scale=_evaluate(node.scale, bindings, cache),
                    size=node.shape,
                )
            )

        elif isinstance(node, graph.uniform_cdf):
            result = stats.uniform.cdf(
                _evaluate(node.x, bindings, cache),
                loc=_evaluate(node.loc, bindings, cache),
                scale=_evaluate(node.scale, bindings, cache),
            )

        elif isinstance(node, graph.normal_rvs):
            result = np.array(
                stats.norm.rvs(
                    loc=_evaluate(node.loc, bindings, cache),
                    scale=_evaluate(node.scale, bindings, cache),
                    size=node.shape,
                )
            )

        elif isinstance(node, graph.normal_cdf):
            result = stats.norm.cdf(
                _evaluate(node.x, bindings, cache),
                loc=_evaluate(node.loc, bindings, cache),
                scale=_evaluate(node.scale, bindings, cache),
            )

        elif isinstance(node, graph.maximum):
            result = np.maximum(
                _evaluate(node.x, bindings, cache), _evaluate(node.y, bindings, cache)
            )

        elif isinstance(node, graph.reduce_mean):
            result = np.mean(_evaluate(node.x, bindings, cache), axis=node.axis)

        elif isinstance(node, graph.exp):
            result = np.exp(_evaluate(node.x, bindings, cache))

        elif isinstance(node, (graph.power, graph.pow)):
            result = np.power(
                _evaluate(node.x, bindings, cache), _evaluate(node.y, bindings, cache)
            )

        elif isinstance(node, graph.log):
            result = np.log(_evaluate(node.x, bindings, cache))

        else:
            raise TypeError(f"Unknown node type: {type(node)}")

    except Exception as e:
        if debug and node.name:
            print(f"→ Evaluation failed: {str(e)}")
            print(f"--- end eval trace (failed) ---")
        raise

    # Post-evaluation debug info
    if debug and node.name:
        print(f"→ Evaluation succeeded")
        print(f"→ Result shape: {result.shape}")
        print(f"→ Result:\n{result}")
        print(f"--- end eval trace (success) ---")

    cache.put(node, result)
    return result


# --- Public API ---


@runtime_checkable
class TensorGraph(Protocol):
    """TensorGraph is the protocol allows to access all the tensors within a
    tensor graph in the order in which they were declared."""

    def iterable_graph(self) -> Iterator[graph.Tensor]: ...


class PartialEvaluationContext:
    """Specific evaluation context bound to:

    1. A set of bindings for placeholders (empty by default).

    2. A cache for memoizing intermediate results (NullCache by default).

    3. Whether to enable debug output (false by default).

    The evaluation context is "partial" in that some tensors may not
    have been evaluated yet. Without caching, no tensor is ever in the
    evaluated state. With caching, we know the value of all the
    already-evaluated tensors.
    """

    def __init__(
        self,
        *,
        cache: Cache | None = None,
        debug: bool = False,
        bindings: Bindings | None = None,
    ):
        self.bindings = bindings or {}
        self.cache = cache or NullCache()
        self.debug = debug

    def evaluate(self, tensor: graph.Tensor) -> Array:
        """
        Evaluates a specific tensor. Depending on the caching policy
        tensors may be evaluated each time or just once.
        """
        return _evaluate(tensor, self.bindings, self.cache, self.debug)

    def evaluate_iterator(self, tensors: Iterator[graph.Tensor]):
        """Like evaluate but operates on an iterator on tensors."""
        for tensor in tensors:
            self.evaluate(tensor)

    def evaluate_graph(self, graph: TensorGraph):
        """Like evaluate but operates on a tensor graph."""
        self.evaluate_iterator(graph.iterable_graph())

    def get_cached(self, tensor: graph.Tensor) -> Array | None:
        """Get cached value without computation."""
        return self.cache.get(tensor)

    def is_cached(self, tensor: graph.Tensor) -> bool:
        """Check if tensor result is in cache."""
        return self.get_cached(tensor) is not None


def evaluate(
    tensor: graph.Tensor, bindings: Bindings, cache: Cache, debug: bool = False
) -> Array:
    """One-shot tensor evaluation with specific bindings and a given caching
    policy as well as optional debugging output. Note that, when using the
    DictCache policy, each intermediate tensor will be evaluated at most once."""
    return _evaluate(tensor, bindings, cache, debug)
