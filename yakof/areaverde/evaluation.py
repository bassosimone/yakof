
import numpy as np
import pandas as pd
import matplotlib.dates as mdates

from vehicle_stats import vehicle_inflow, vehicle_starting
from yakof.frontend import linearize, graph
from yakof.areaverde.indexes import *
from yakof.numpybackend import executor
from yakof.dtyak import Index
from matplotlib.ticker import FuncFormatter
from matplotlib import pyplot as plt
from scipy import stats

# indexes needed to apply the TS_sum function
to_sum_indexes = [I_total_anticipating, I_total_postponing,
                  I_total_base_flow, I_total_reduced_flow,
                  I_total_paying, I_total_anticipated,
                  I_total_postponed, I_total_reduced_emissions, I_total_payed, I_total_emissions]

def TS_sum(ts: np.ndarray) -> np.ndarray:
    return np.array([[ts.sum()]])


def TS_solve(ts: np.ndarray, total_traffic: float = 1684783, max_iter: int = 50) -> np.ndarray:
    # TODO: total_traffic should not be a constant value
    series = ts.copy()
    for _ in range(max_iter):  # TODO: decide when to finish based on convergence?
        mu = 1 + 3 * sum(series) / total_traffic
        alpha = (mu - 1) / mu
        series = ts + np.roll(series, 1, axis=0) * alpha
    return series

def linearize_indexes(index_list: list[Index])->list[graph.Node]:
    """Linearizes a list of index nodes into an ordered sequence for evaluation."""
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

    initial_state = executor.State(
    values={
        TS.node: np.array(
            [(t - pd.Timestamp("00:00:00")).total_seconds() for t in 
             pd.date_range(start="00:00:00", periods=12 * 24, freq="5min")]
        ),
        TS_inflow.node: vehicle_inflow,                                                     
        TS_starting.node: vehicle_starting,
        I_B_p50_cost.node: np.array([UniformDistIndex('cost 50% threshold', loc=4.00, scale=7.00).value.rvs()])
    },
)
    evaluated_results = {}
    for node in linearized_nodes:
        result = executor.evaluate(state=initial_state, node=node)
        evaluated_results[node.name] = result

    
    for index in to_sum_indexes:
        evaluated_results[index.name] = TS_sum(evaluated_results[index.name])

    for index in [I_traffic, I_reduced_traffic]:
        evaluated_results[index.name] = TS_solve(evaluated_results[index.name])
    

    return evaluated_results




def distribution(field, size=10000, num=100):
    xx, yy = np.meshgrid(np.linspace(0, size, num + 1), range(field.shape[1]))
    zz = stats.poisson(mu=np.expand_dims(field, axis=2)).cdf(np.expand_dims(xx, axis=0))
    return zz.mean(axis=0)


def plot_field_graph(field, horizontal_label, vertical_label, vertical_size, vertical_formatter=None,
                     reference_line=None):
    dist = distribution(field, vertical_size, 100)
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(pd.date_range(start='00:00:00', periods=12 * 24, freq='5min'),
                   np.linspace(0, vertical_size, 100 + 1), dist.T,
                   cmap='coolwarm_r', vmin=0.0, vmax=1.0)
    if reference_line is not None:
        plt.plot(pd.date_range(start='00:00:00', periods=12 * 24, freq='5min'),
                 reference_line, linewidth=2, color='black')
    plt.plot(pd.date_range(start='00:00:00', periods=12 * 24, freq='5min'),
             field.mean(axis=0), linewidth=1, color='black')
    plt.gca().set_ylim([0, vertical_size])
    if vertical_formatter is not None:
        plt.gca().yaxis.set_major_formatter(vertical_formatter)
    plt.gca().set_ylabel(vertical_label)
    plt.gcf().tight_layout()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gcf().autofmt_xdate()
    plt.gca().set_xlabel(horizontal_label)


def compute_kpis(evals):
    return {
        'Base flow': int(evals[I_total_base_flow].mean()),
        'Reduced flow': int(evals[I_total_reduced_flow].mean()),
        'Shifted flow': int(evals[I_total_shifted].mean()),
        'Paying flow': int(evals[I_total_paying].mean()) if evals[I_avg_cost].mean() > 0 else 0,
        'Collected fees': int(evals[I_total_payed].mean()),
        'Reduced emissions (NOx gr/day)': int(evals[I_total_emissions].mean()) - int(evals[I_total_reduced_emissions].mean())
    }


if __name__ == "__main__":
    results = evaluate_indexes(indexes)
    print(results)
    # plot_field_graph(results[I_reduced_flow],
    #                  horizontal_label="Time", vertical_label="Flow (vehicles/hour)",
    #                  vertical_size=1250,
    #                  vertical_formatter=FuncFormatter(lambda x, _: f"{int(x * 12)}"),
    #                  reference_line=results[TS_inflow][0])
    # plt.show()

    # plot_field_graph(results[I_reduced_traffic],
    #                  horizontal_label="Time", vertical_label="Traffic (circulating vehicles)",
    #                  vertical_size=15000,
    #                  reference_line=results[I_traffic][0])
    # plt.show()

    # plot_field_graph(results[I_reduced_emissions],
    #                  horizontal_label="Time", vertical_label="Emissions (NOx g/h)",
    #                  vertical_size=3000,
    #                  vertical_formatter=FuncFormatter(lambda x, _: f"{int(x * 12)}"),
    #                  reference_line=results[I_emissions][0])
    # plt.show()

    # for k, v in compute_kpis(results).items():
    #     print(f'{k} - {v:,}')