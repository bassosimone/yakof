"""
...
"""

from dataclasses import dataclass
from ..frontend import graph

import threading


@dataclass(frozen=True)
class SymbolValue:
    node: graph.placeholder
    value: str


class _SymbolTable:
    def __init__(self):
        self._table: dict[str, SymbolValue] = {}
        self._lock = threading.Lock()

    def get(self, name: str):
        with self._lock:
            if name not in self._table:
                self._table[name] = SymbolValue(graph.placeholder(name), name)
            return self._table[name]

    def values(self) -> list[SymbolValue]:
        with self._lock:
            values = list(self._table.values())
        return values


symbol_table = _SymbolTable()


def Symbol(name: str) -> SymbolValue:
    return symbol_table.get(name)
