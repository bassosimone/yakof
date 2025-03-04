from typing import Protocol, Sequence, runtime_checkable

import random

from ..frontend import autoenum, graph

from . import geometry


Weight = float

EnumValue = float


@runtime_checkable
class GeneralizedDistribution(Protocol):
    def support_size(self) -> int: ...

    def sample(
        self,
        nr: int = 1,
        *,
        subset: Sequence[str] | None = None,
        force_sample: bool = False,
    ) -> list[tuple[Weight, EnumValue]]: ...


class _CategoricalDistr:
    def __init__(self, values: dict[str, int]) -> None:
        self.values = values

    def support_size(self) -> int:
        return len(self.values)

    def sample(
        self,
        nr: int = 1,
        *,
        subset: Sequence[str] | None = None,
        force_sample: bool = False,
    ) -> list[tuple[Weight, EnumValue]]:
        # TODO: subset (if defined) should be a subset of the support (also: with repetitions?)

        keys, size = list(self.values.keys()), len(self.values)
        if subset is not None:
            keys, size = subset, len(subset)

        if force_sample or nr < size:
            ratio = 1 / nr
            keys = random.choices(keys, k=nr)
        else:
            ratio = 1 / size
            keys = keys

        return [(ratio, self.values[key]) for key in keys]


class Variable(geometry.ComputationTensor):
    def __init__(self, node: graph.Node, distribution: GeneralizedDistribution):
        super().__init__(geometry.ComputationSpace, node)
        self.distribution = distribution


class _CategoricalVariable(Variable):
    def __init__(self, node: graph.Node, names: dict[str, int]) -> None:
        super().__init__(node, _CategoricalDistr(names))
        self.names = names

    def __hash__(self):
        return id(self)

    def __eq__(self, other: str) -> geometry.ComputationTensor:  # type: ignore
        return geometry.ComputationSpace.equal(
            self,
            geometry.ComputationSpace.constant(self.names[other]),
        )

    # FIXME: this method should actually be deprecated
    def sample(
        self,
        nr: int = 1,
        *,
        subset: Sequence[str] | None = None,
        force_sample: bool = False,
    ) -> list[tuple[Weight, EnumValue]]:
        return self.distribution.sample(nr, subset=subset, force_sample=force_sample)


def categorical_variable(name: str, values: dict[str, float]) -> Variable:
    assert len(values) > 0
    enumcore = autoenum.Type(geometry.ComputationSpace, name)
    name_to_enum_value = {value: autoenum.Value(enumcore, value) for value in values}
    name_to_id = {
        name: enumvalue.value for name, enumvalue in name_to_enum_value.items()
    }
    return _CategoricalVariable(enumcore.tensor.node, name_to_id)


def uniform_categorical_variable(name: str, values: Sequence[str]) -> Variable:
    assert len(values) > 0
    return categorical_variable(name, {value: 1 / len(values) for value in values})
