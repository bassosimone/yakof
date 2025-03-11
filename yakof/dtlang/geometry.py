"""
Model Geometry
==============

This module defines the model geometry in which we operate. We use a 3d
space where the first two dimensions are presence variables and the third
dimension contains the ensemble space.
"""

from ..frontend import abstract, bases

space = abstract.TensorSpace(bases.XYZ())
"""Computation space instance."""

Tensor = abstract.Tensor[bases.XYZ]
"""Computation tensor type."""
