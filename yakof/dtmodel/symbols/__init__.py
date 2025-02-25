"""
Digital Twin Model
==================

This package is a fork of `dt_model` (https://github.com/fbk-most/dt_model)
using `yakof` as the computation engine. The purpose of this package is to
explore using `yakof` with `dt_model` in a sandbox environment. We will eventually
merge these changes back into the `dt_model` proper.

Integration Notes
-----------------

This section serves as a diary. We are keeping track of the process with
which we imported existing `dt-model` sources and modified them to use yakof
rather than using the underlying `sympy` library;

1. 2025-02-25: imported `index.py` and replaced the `SymbolExtender`
dependency with the `yakof.symbolic.symbol.Symbol` class. Noted that the
`index.py` file depends on the context variable, so I proceed to also
import `context_variable.py` into this package. Noted that we still have
a residual dependency on `lambdify`, which has not been dealt with yet.

2. 2025-02-25: imported `context_variable.py` and replaced the `SymbolExtender`
with the `yakof.symbolic.symbol.Symbol` class. Noted that `ContextVariable`
is actually an abstract base class, so used `abc` to quell squiggles. After
this change, there is no residual dependency on `sympy`, so we are kind of good
except for a `self.rvc.rvs` squiggle to further investigate. I decided to fix
this using `#type: ignore` command for the time being.

3. 2025-02-25: knowing I still need to address the `lambidy`, I am nonetheless
choosing to focus on the `constraint.py` file. This file is probably
easy to import, so it looks like a good place to start and make some progress. I
changed the import from `sympy` to `symbolic` and nothing else.

4. 2025-02-25: I now need to deal with `presence_variable.py`. I replaced the
imports and then I deployed a type assertion that I will need to check later
regarding `self.distribution` not being `None`. I also needed a `type: ignore`
because the return value may not be a numpy array.

5. 2025-02-25: now it's time to circle back to `index.py` and see how we
can ploy to avoid using lambdify directly. I start by constraining the type
passed to the `Index` constructor from `Any` to become instead a
`Distribution|int|float|Symbol`. Most of the `SymIndex` constructor
is also redundant considering the `Index` constructor. I modified the
`Index` to store into `.value` a reference to itself as a `graph.Node` and
would need to modify the evaluation loop to (a) use a yakof cache and (b)
use the `evaluator` package to evaluate to a numpy array. I think this also
raises the question of whether we want any `.rvs` stuff inside the graph.

6. 2025-02-25: [outside of this package but this does not matter since
these notes are not meant to survive this experiment] I have also
imported `model.py` into a dedicated package called `model`. I have
added `type: ignore` below a specific point in the file to silence the
squiggles that are not related to the yakof integration. Beyond this,
I restructured the `evaluate` loop and turned the `Ensemble` into
an interface so that we can cut at the complexity boundary.

7. 2025-02-25: While rewriting the `Model` loop I realize that I have
probably done something wrong with the `Constraint.capacity` field: the
type should either be a `Symbol` or a `Distribution`. At this point,
I finished adapting the code in the `Model` loop.
"""
