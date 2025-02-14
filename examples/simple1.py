"""
Demonstrates:

    1. using the tensors frontend to define a simple model

    2. automatic tensor naming using context managers

    3. grid evaluation using the NumPy HIR evaluator
"""

# TODO(bassosimone): adapt the concept of automatic disjoint enum
# generation from the phasespace package.


from yakof.frontend import abstract, autonaming, bases, pretty
from yakof.numpybackend import hir, evaluator

import numpy as np

# We define a tensor space using as basis the default X asis (in other
# words, we're operating in R^1 here and we only need a single axis)
space = abstract.TensorSpace(bases.X)

# We use the autonaming context to automatically name the tensors
with autonaming.context():
    sitting = space.placeholder()
    takeaway = space.placeholder()
    seats = space.constant(50)
    sustainable = sitting + takeaway <= seats

# We use the pretty printer to print the sustainability tensor
print(pretty.format(sustainable.t))

# We transform the AST into a NumPy HIR
sustainable_ir = hir.transform(sustainable.t)

# We evaluate the HIR using the NumPy evaluator
xx, yy = np.meshgrid(np.linspace(0, 100, 10), np.linspace(0, 100, 10))
result = evaluator.evaluate(sustainable_ir, {
    "sitting": xx,
    "takeaway": yy,
})
print(result)
