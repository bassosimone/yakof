"""
Demonstrates:

    1. using the NumPy backend directly to define a model

    2. could be the blueprint for numeric evaluation blocks
"""

from yakof.numpybackend import hir, evaluator

import numpy as np

# Directly define the model in terms of the NumPy semantic tree
sitting = hir.placeholder("sitting")
takeaway = hir.placeholder("takeaway")
seats = hir.constant(np.array(50))
sustainable = hir.less_equal(hir.add(sitting, takeaway), seats)

# Evaluate the model with the HIR evaluator
xx, yy = np.meshgrid(np.linspace(0, 100, 10), np.linspace(0, 100, 10))
result = evaluator.evaluate(sustainable, {
    "sitting": xx,
    "takeaway": yy,
})
print(result)
