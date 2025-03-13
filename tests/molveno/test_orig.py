"""Tests for the yakof.molveno.orig module."""

# SPDX-License-Identifier: Apache-2.0

from dt_model import Ensemble
from yakof.molveno import orig as mmod

import numpy as np


def test_molveno_orig_simple():
    """Simple test ensuring we can evaluate M_base."""

    # Load the base model
    model = mmod.M_Base

    # Define the required extra variables to evaluate the model
    ensemble_size = 20
    (t_max, e_max) = (10000, 10000)
    (t_sample, e_sample) = (100, 100)
    situation = mmod.S_Base

    ensemble = Ensemble(model, situation, cv_ensemble_size=ensemble_size)
    tt = np.linspace(0, t_max, t_sample + 1)
    ee = np.linspace(0, e_max, e_sample + 1)
    xx, yy = np.meshgrid(tt, ee)
    zz = model.evaluate({mmod.PV_tourists: tt, mmod.PV_excursionists: ee}, ensemble)
