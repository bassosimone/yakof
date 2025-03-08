"""
Digital Twins Language (dtlang)
===============================

This package provides a domain-specific language for creating and evaluating
digital twins sustainability models. It integrates concepts from:

- Tensor computations in a 3D XYZ space
- Context variables representing uncertainty
- Presence variables defining spatial dimensions
- Constraints between resource usage and capacity
- Ensemble-based evaluation to account for uncertainty

The package allows for definition and evaluation of sustainability models
that can assess environmental constraints across spatial grids while
incorporating both deterministic and probabilistic elements.

Key components:
- Model: The main container for all model elements
- Constraint: Relates resource usage to capacity
- PresenceVariable: Defines spatial dimensions
- ContextVariable: Represents sources of uncertainty
- Index: Parameters that may be constant or sampled from distributions
- Piecewise: Support for conditional expressions in models
"""

from .constraint import Constraint
from .context import UniformCategoricalContextVariable
from .index import Index
from .model import Model
from .piecewise import Piecewise
from .presence import PresenceVariable

__all__ = [
    "Constraint",
    "Index",
    "Model",
    "Piecewise",
    "PresenceVariable",
    "UniformCategoricalContextVariable",
]
