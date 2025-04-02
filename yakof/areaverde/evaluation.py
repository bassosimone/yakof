import datetime
import pickle
import numpy as np
import pandas as pd
import json 
from pprint import pprint

from yakof.frontend import pretty
from yakof.frontend import linearize, graph
from yakof.areaverde.areaverde_yak import *
from yakof.numpybackend import executor
from yakof.dtyak import UniformDistIndex, Index
from vehicle_stats import vehicle_inflow, vehicle_starting

def TS_sum(ts: np.ndarray) -> np.ndarray:
    return np.expand_dims(sum(ts), axis=1)   

def TS_solve(ts: np.ndarray, total_traffic: float = 1684783, max_iter: int = 50) -> np.ndarray:
    # TODO: total_traffic should not be a constant value
    # print(ts)
    series = ts.copy()
    for _ in range(max_iter):  # TODO: decide when to finish based on convergence?
        mu = 1 + 3 * sum(series) / total_traffic
        alpha = (mu     - 1) / mu
        series = ts + np.roll(series, 1, axis=0) * alpha
    # print(series)
    return series

def linearize_indexes(index_list: list[Index])->list[graph.Node]:
    """Linearizes a list of index nodes into an ordered sequence for evaluation."""
    for index in index_list:
        print(index.node.name
              )
    plan = linearize.forest(*[index.node for index in index_list])
    # for node in plan:
    #     print(node.name)
        # print(node)
    return linearize.forest(*[index.node for index in index_list])


def evaluate_indexes(index_list: list[Index])-> dict:
    """
    Evaluates a set of index nodes by executing them in topological order.

    Args:
        index_list: A list of index nodes to evaluate.

    Returns:
        Dictionary of evaluated results.
    """
    linearized_nodes = linearize_indexes(index_list)
    # pprint(linearized_nodes)
    initial_state = executor.State(
        values={
            TS.node: np.array(
                [(t - pd.Timestamp("00:00:00")).total_seconds() for t in 
                 pd.date_range(start="00:00:00", periods=12 * 24, freq="5min")]
            ),
            TS_inflow.node: vehicle_inflow,     
            TS_starting.node: vehicle_starting,
            I_B_p50_cost.node: np.array(
                6.6 # just to get same result
            )
        },
        # flags=graph.NODE_FLAG_TRACE
    )
    evaluated_results = {}
    for node in linearized_nodes:
        # print(pretty.format(node))
        result = executor.evaluate(state=initial_state, node=node)
        # print(result)
        evaluated_results[node.name] = result
    # print(evaluated_results[I_emissions])

    return evaluated_results


# Run evaluation
results = evaluate_indexes([I_total_emissions])


# # ‌ TS_sum
# to_sum_indexes = [I_total_anticipating, I_total_postponing,
#                     I_total_base_flow,I_total_reduced_flow,
#                     I_total_paying,I_total_anticipated,
#                     I_total_postponed, I_total_reduced_emissions, I_total_payed, I_total_emissions]

# # **Optimized TS_sum using NumPy vectorized sum**
# results.update({index.name: np.sum(results[index.name]) for index in to_sum_indexes})

# # **Efficient TS_solve computation**
# for index in [I_traffic, I_reduced_traffic]:
#     results[index.name] = TS_solve(results[index.name])

# pprint(results)


