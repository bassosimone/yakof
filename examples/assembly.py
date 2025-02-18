"""
Shows what IMHO is the structure of the computation used by the dt-model
when modeling the Molveno lake. More specifically:

1. We have independent presence variables aligned along X and Y.

2. The ensemble dimension is aligned along Z and consists of several parallel
vectors. We have N+1 vectors, where N is the number of context variables and the
additional vector provides weights to aggregate the results.

I called this file assembly.py because it feels very low level. On the plus
side, the actual structure of the computation is explicit. On the negative side,
however, one needs to write down each operation explicitly. In the event in
which we have several similar models, this could become a drawback. Thus, to me,
this file seems a stepping stone towards understanding better abstractions.

Perhaps, a good development direction would be to understand how to generate
alike code starting from the data structured exported by `dt-model`.

Regardless, what I find particularly useful of this representation is that it is
possible to abstract the whole computation in terms of:

1. the tensor or tensors to evaluate (we have caching so we don't need to
worry about recomputing part of the computation graph);

2. a dictionary mapping placeholders to their actual values.

All in all, this provides quite robust separation of concerns.
"""

from yakof.frontend import abstract, autoenum, autonaming, bases, spaces
from yakof.numpybackend import evaluator

import numpy as np

enumspace = autoenum.Space()

weather_good, weather_bad = enumspace.define_enum("weather", "good", "bad")

# Automatically assign names to tensors
with autonaming.context():
    # Define presence variables
    tourists = spaces.x.placeholder()
    excursionists = spaces.y.placeholder()

    # Define context variables
    weights = spaces.z.placeholder()
    weather = spaces.z.placeholder()

    # Define weather dependent capacity
    capacity_good_weather = spaces.z.placeholder(default_value=100.0)
    capacity_bad_weather = spaces.z.placeholder(default_value=30.0)
    capacity = spaces.z.where(
        weather == weather_good.value,
        capacity_good_weather,
        capacity_bad_weather,
    )

    # Define weather dependent usage factors
    usage_factor_good_weather = spaces.z.placeholder(default_value=1.0)
    usage_factor_bad_weather = spaces.z.placeholder(default_value=0.5)
    usage_factor = spaces.z.where(
        weather == weather_good.value,
        usage_factor_good_weather,
        usage_factor_bad_weather,
    )

    # Expand to the XYZ space to perform the computation
    tourists_xyz = bases.expand_xy_to_xyz(bases.expand_x_to_xy(tourists))
    excursionists_xyz = bases.expand_xy_to_xyz(bases.expand_y_to_xy(excursionists))
    usage_factor_xyz = bases.expand_yz_to_xyz(bases.expand_z_to_yz(usage_factor))
    capacity_xyz = bases.expand_yz_to_xyz(bases.expand_z_to_yz(capacity))
    weights_xyz = bases.expand_xz_to_xyz(bases.expand_z_to_xz(weights))

    # Define overall usage
    total_usage_xyz = (
        tourists_xyz * usage_factor_xyz + excursionists_xyz * usage_factor_xyz
    )

    # Estimate sustainability
    sustainable_xyz = total_usage_xyz <= capacity_xyz

    # Apply weights
    sustainable_weighted_xyz = sustainable_xyz * weights_xyz

    # Sum over the ensemble axis
    sustainable = bases.project_xyz_to_xy_using_sum(sustainable_weighted_xyz)

# Generate ensemble values
weather_ensemble = np.array([weather_good.value, weather_bad.value])
weights_ensemble = np.array([0.65, 0.35])

# Generate presence variable ranges
tourist_range = np.linspace(0, 100, 5)
excursionist_range = np.linspace(0, 100, 5)

# Evaluate with explicit bindings
result = evaluator.evaluate(
    sustainable.node,
    {
        "tourists": tourist_range,
        "excursionists": excursionist_range,
        "weather": weather_ensemble,
        "weights": weights_ensemble,
        "capacity_good_weather": np.asarray(100.0),
        "capacity_bad_weather": np.asarray(30.0),
        "usage_factor_good_weather": np.asarray(1.0),
        "usage_factor_bad_weather": np.asarray(0.5),
    },
)

print(result)
