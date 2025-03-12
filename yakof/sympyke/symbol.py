"""
...
"""

from ..frontend import graph

import threading


class _SymbolTable:
    def __init__(self):
        self._table: dict[str, graph.constant] = {}
        self._lock = threading.Lock()
        self._id = 0

    def get(self, name: str):
        with self._lock:
            if name not in self._table:
                id = self._id
                self._id = self._id + 1
                self._table[name] = graph.constant(id)
            return self._table[name]


_symbol_singleton = _SymbolTable()


def Symbol(name: str) -> graph.constant:
    return _symbol_singleton.get(name)
