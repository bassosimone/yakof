"""Runnable definition of the original model."""

# SPDX-License-Identifier: Apache-2.0

from dt_model import Ensemble, Index, Model
from yakof.molveno import orig as mmod

import numpy as np
import random


def reset_and_run() -> Model:
    """Simple test ensuring we can evaluate M_base."""

    # Load the base model
    model = mmod.M_Base

    # Ensure we reset the model to evaluate it again
    #
    # TODO(bassosimone): the model should not keep state
    model.reset()

    # Define the required extra variables to evaluate the model
    ensemble_size = 1000
    (t_max, e_max) = (10000, 10000)
    (t_sample, e_sample) = (100, 100)
    situation = mmod.S_Base

    # Force the same random seed
    random.seed(4)
    np.random.seed(4)

    # Evaluate the model
    ensemble = Ensemble(model, situation, cv_ensemble_size=ensemble_size)
    tt = np.linspace(0, t_max, t_sample + 1)
    ee = np.linspace(0, e_max, e_sample + 1)
    xx, yy = np.meshgrid(tt, ee)
    zz = model.evaluate({mmod.PV_tourists: tt, mmod.PV_excursionists: ee}, ensemble)
    return model
