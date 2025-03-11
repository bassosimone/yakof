"""Tests for the yakof.dtlang.presence module."""

# SPDX-License-Identifier: Apache-2.0

from yakof.dtlang import context, presence


def test_presence_variable():
    """Ensure that the PresenceVariable is working as intended."""

    # Create the context variables on which we depend
    cv1 = context.UniformCategoricalContextVariable("cv1", ["a", "b", "c"])
    cv2 = context.UniformCategoricalContextVariable("cv1", ["a", "b", "c"])

    # Create the actual presence variable
    pv = presence.PresenceVariable("pv", [cv1, cv2])

    # Make sure the name is correct
    assert pv.name == "pv"

    # Make sure the list of context variables is correct
    assert pv.cvs == [cv1, cv2]
