import numpy as np

from yakof.frontend import graph
from yakof.numpybackend import executor
from yakof import trafficmodel


# Build model
inputs = trafficmodel.Inputs()
model = trafficmodel.build(inputs)

# Set up state with initial placeholders
state = executor.State(
    values={
        inputs.base_demand.node: np.array([10.0, 20.0, 50.0, 30.0, 15.0, 10.0, 10.0]),
        inputs.price.node: np.array([1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0]),
        inputs.hours.node: np.array([6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        inputs.price_sensitivity.node: np.array([0.05, 0.1, 0.15]),
    },
    flags=graph.NODE_FLAG_TRACE,
)

# Evaluate the model
for node in model.nodes:
    executor.evaluate(state, node)
