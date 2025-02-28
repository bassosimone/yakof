"""
Dispatch Operations
===================

...
"""

import numpy as np

from ..frontend import graph

# This dispatch table maps a binary op in the graph domain
# to the corresponding numpy operation. Add to this table to
# add support for more binary operations.
binary_operations = {
    graph.add: np.add,
    graph.subtract: np.subtract,
    graph.multiply: np.multiply,
    graph.divide: np.divide,
    graph.equal: np.equal,
    graph.not_equal: np.not_equal,
    graph.less: np.less,
    graph.less_equal: np.less_equal,
    graph.greater: np.greater,
    graph.greater_equal: np.greater_equal,
    graph.logical_and: np.logical_and,
    graph.logical_or: np.logical_or,
    graph.logical_xor: np.logical_xor,
    graph.power: np.power,
    graph.maximum: np.maximum,
}


# Like binary_operations but for unary operations
unary_operations = {
    graph.logical_not: np.logical_not,
    graph.exp: np.exp,
    graph.log: np.log,
}


def __expand_dims(x: np.ndarray, axis: graph.Axis) -> np.ndarray:
    """Internal expand_dims implementation used by axis_ops_dispatch_table"""
    return np.expand_dims(x, axis)


def __reduce_sum(x: np.ndarray, axis: graph.Axis) -> np.ndarray:
    """Internal reduce_sum implementation used by axis_ops_dispatch_table"""
    return np.sum(x, axis=axis)


def __reduce_mean(x: np.ndarray, axis: graph.Axis) -> np.ndarray:
    """Internal reduce_mean implementation used by axis_ops_dispatch_table"""
    return np.mean(x, axis=axis)


# Like binary_operations but for axis operations
axes_operations = {
    graph.expand_dims: __expand_dims,
    graph.reduce_sum: __reduce_sum,
    graph.reduce_mean: __reduce_mean,
}
