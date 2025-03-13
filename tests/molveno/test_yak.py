"""Tests for the yakof.molveno.orig module."""

# SPDX-License-Identifier: Apache-2.0

from .import yak


def test_molveno_yak_simple():
    """Simple test ensuring we can evaluate M_base."""
    _ = yak.reset_and_run()
