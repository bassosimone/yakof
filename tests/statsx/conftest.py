"""Test fixtures for yakof.statsx package."""

# SPDX-License-Identifier: Apache-2.0

import pytest
import random
import numpy as np


@pytest.fixture
def fixed_random_seed():
    """Set fixed seeds for all random generators."""
    # Save the original states
    py_state = random.getstate()
    np_state = np.random.get_state()

    # Set seeds
    random.seed(42)
    np.random.seed(42)

    # Provide the fixture
    yield

    # Restore the original states
    random.setstate(py_state)
    np.random.set_state(np_state)
