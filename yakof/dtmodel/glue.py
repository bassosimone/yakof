from typing import Sequence

from dt_model import UniformCategoricalContextVariable
import numpy as np

from . import (
    constraint,
    context,
    ensemble,
    evaluator,
    index,
    model,
    piecewise,
    presence,
)

CategoricalContextVariable = context.categorical_variable

ContextVariable = context.Variable

Constraint = constraint.Expression

Index = index.construct

Piecewise = piecewise.to_tensor

PresenceVariable = presence.construct

UniformCategoricalContextVariable = context.uniform_categorical_variable


class Model(model.Model):
    def evaluate(
        self,
        grid: dict[presence.Variable, np.ndarray],
        ensemble: ensemble.Iterator,
    ):
        return evaluator.evaluate(self, grid, ensemble)
