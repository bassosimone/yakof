"""Tests for the yakof.molveno.orig module."""

# SPDX-License-Identifier: Apache-2.0

from .import orig


def test_molveno_orig_simple():
    """Simple test ensuring we can evaluate M_base."""
    _ = orig.reset_and_run()
