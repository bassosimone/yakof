"""
Presence Variables
==================

This module defines presence variables which represent the spatial dimensions
in the model's grid. These variables are used to create the coordinate system
within which sustainability assessments are performed.

Presence variables can depend on context variables, allowing the spatial
dimensions to be influenced by uncertainty factors.
"""

from typing import Sequence

from ..frontend import graph

from . import geometry
from .context import ContextVariable


class PresenceVariable(geometry.Tensor):
    """Represents a spatial dimension in the sustainability grid.

    A presence variable defines one axis of the spatial grid used for
    sustainability assessment. It can depend on context variables,
    allowing the spatial dimension to be influenced by uncertainty.

    Args:
        name: The name of the presence variable
        cvs_deps: Sequence of context variables that this presence variable depends on
    """

    def __init__(self, name: str, cvs_deps: Sequence[ContextVariable]) -> None:
        super().__init__(geometry.space, graph.placeholder(name))
        self.cvs = cvs_deps
