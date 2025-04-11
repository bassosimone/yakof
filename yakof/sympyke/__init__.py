"""
SymPy compatibility layer.
=========================

This package implements a tiny SymPy compatibility layer that maps
sympy-like methods and functions to frontend.graph operations.
"""

from .operators import Eq
from .piecewise import Piecewise
from .symbol import Symbol

__all__ = ["Eq", "Piecewise", "Symbol"]
