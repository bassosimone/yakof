"""
SymPy compatibility layer
=========================

This package implements a SymPy compatibility layer. The purpose
of this layer is to facilitate upgrading the `dt-model` repository
(https://github.com/fbk-most/dt-model) from using sympy as the
underlying evaluation engine to using yakof with a numpy backend.

Integration Plan
----------------

The rough idea to perform the integration is to create a clone
of the `dt-model` in this repository and then rewrite it in
terms of this package, trying to preserve the original structure
as much as possible. Once this has been done, we will run
model specific integration tests to ensure everything is still
working as intended. When this is successful, we will then
proceed with merging changes back into `dt-model`.
"""

# SPDX-License-Identifier: Apache-2.0
