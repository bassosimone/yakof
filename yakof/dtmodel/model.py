from dataclasses import dataclass, field
from typing import Sequence

from . import constraint, context, geometry, index, presence


@dataclass(frozen=True)
class Model:
    name: str
    cvs: Sequence[context.Variable]
    pvs: Sequence[presence.Variable]
    indexes: Sequence[index.Placeholder | geometry.ComputationTensor]
    capacities: Sequence[index.Placeholder]
    constraints: Sequence[constraint.Expression]
