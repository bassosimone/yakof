"""
Computation backend
===================

The computation backend allows to represent digital twins sustainability
models as an internal domain specific language (DSL). Such a DSL uses
names and concepts borrowed from TensorFlow, which, in turns, is inspired
by names and concepts introduced by NumPy.

The reason why we use a DSL is that we want to decouple the equations
representing a sustainability model from their evaluation. This design
choice allows for the following beneficial properties:

1. Data scientists could write the set of equations using a familiar
vocabulary (i.e., the one of NumPy and TensorFlow).

2. The actual evaluation could use distinct evaluation engines, including
by default NumPy, but possibly TensorFlow in case we need GPUs.

3. In a similar fashion, the model could be evaluated using multiple
machines and CPUs, using Dask or other distributed libraries.

A previous version of this module used SymPy for symbolic computation. Yet, we
found that rolling our own DSL allowed us to more easily integrate with NumPy
concepts, with benefits such as automated handling of dimensions. Additionally,
this DSL does not have the performance penalty caused by sympy.lambdify, even
though there is a penalty caused by evaluating the DSL's AST. However, because
the bottleneck is numerical computation, this overhead seems to be small
compared to the flexibility benefits we gained.

Overall, the goal of this module is to provide a solid low-level foundation
for writing sustainability models at low-level. Also, higher-level modules should
actually be used to simplify our job. We are aiming to avoid putting too many
conceptual branches inside the model description, and we are trying instead to
put this kind of complexity inside the scheduling of the model.

That said, there are some conditional operations, which may be useful when
writing more simplified models. For example, we have a `where` operation and
a `multi_clause_where` operation (somehow similar to `numpy.select`).

The overall approach with which we are writing this module is deeply rooted
into New Jersey style programming and the YAGNI principle. Therefore, we tried
to design a reasonably regular architecture that can be extended easily on
demand, without implementing functions we don not need at the moment.

The module is organized as follows:

1. `graph` contains the code to build an abstract computation graph.

2. `engine` contains the code for instantiating specific computation engines.

3. `numpy_engine` implements evaluation using NumPy.

4. `tensorflow_engine` implements evaluation using TensorFlow.

5. `oriented.py` contains oriented tensors (which helps to ensure
one can only operate on homogeneus tensors).

5. `pretty.py` implements pretty-formatting the computation graph.

SPDX-License-Identifier: Apache-2.0
"""

from . import graph
from . import engine
from . import numpy_engine
from . import oriented
from . import pretty

__all__ = ["graph", "engine", "numpy_engine", "oriented", "pretty"]
