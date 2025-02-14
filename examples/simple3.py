"""
Similar to simple1.py except that we transform the NumPy HIR into
a register indexed linear form (which makes partial evaluation possible
and allows for straightforward caching).
"""

from yakof.frontend import abstract, autonaming, bases, graph
from yakof.numpybackend import hir, emitter, vm

import numpy as np

# Operate in R^1 using the X axis as basis for the space
space = abstract.TensorSpace(bases.X)

# Automatically name tensors within the model
with autonaming.context():
    sitting = space.placeholder()
    takeaway = space.placeholder()
    seats = space.constant(50)
    sustainable = sitting + takeaway <= seats

# Lower to HIR
sustainable_ir = hir.transform(sustainable.t)

# Lower to register indexed linear form
sustainable_prog = emitter.Program()
sustainable_register = emitter.emit(sustainable_ir, sustainable_prog)

# Instantiate and use the virtual machine
#
# Note: if we can interpret the linear form, we can equally well generate
# a Python script to compile to bytecode for maximum efficiency
vm = vm.VirtualMachine()
xx, yy = np.meshgrid(np.linspace(0, 100, 10), np.linspace(0, 100, 10))
result = vm.execute(sustainable_prog, sustainable_register, {
    "sitting": xx,
    "takeaway": yy,
})

print(result)
