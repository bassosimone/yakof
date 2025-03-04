from typing import Iterator as TypingIterator

from . import context

Weight = float

VariableValue = dict[context.Variable, float]

Iterator = TypingIterator[tuple[Weight, VariableValue]]
