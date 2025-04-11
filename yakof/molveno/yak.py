"""
Yakof-based Molveno Model.
=========================

This is a modified Molveno model, using the `yakof.fronted` and
`yakof.numpybackend` as its underlying execution engine.
"""

# SPDX-License-Identifier: Apache-2.0

from yakof.dtyak import (
    CategoricalContextVariable,
    Constraint,
    Index,
    LognormDistIndex,
    Model,
    PresenceVariable,
    TriangDistIndex,
    UniformCategoricalContextVariable,
    UniformDistIndex,
)
from yakof.sympyke import Eq, Piecewise, Symbol

from .presence_stats import (
    excursionist_presences_stats,
    season,
    tourist_presences_stats,
    weather,
    weekday,
)

# MODEL DEFINITION

# Context variables

CV_weekday = UniformCategoricalContextVariable("weekday", [Symbol(v) for v in weekday])
CV_season = CategoricalContextVariable("season", {Symbol(v): season[v] for v in season.keys()})
CV_weather = CategoricalContextVariable("weather", {Symbol(v): weather[v] for v in weather.keys()})

# Presence variables

PV_tourists = PresenceVariable("tourists", [CV_weekday, CV_season, CV_weather], tourist_presences_stats)
PV_excursionists = PresenceVariable("excursionists", [CV_weekday, CV_season, CV_weather], excursionist_presences_stats)

# Capacity indexes

I_C_parking = UniformDistIndex("parking capacity", loc=350.0, scale=100.0)
I_C_beach = UniformDistIndex("beach capacity", loc=6000.0, scale=1000.0)
I_C_accommodation = LognormDistIndex(
    "accommodation capacity",
    s=0.125,
    loc=0.0,
    scale=5000.0,
)
I_C_food = TriangDistIndex("food service capacity", loc=3000.0, scale=1000.0, c=0.5)

# Usage indexes

I_U_tourists_parking = Index("tourist parking usage factor", 0.02)
I_U_excursionists_parking = Index(
    "excursionist parking usage factor",
    Piecewise((0.55, Eq(CV_weather.node, Symbol("bad"))), (0.80, True)),
    cvs=[CV_weather],
)

I_U_tourists_beach = Index(
    "tourist beach usage factor",
    Piecewise((0.25, Eq(CV_weather.node, Symbol("bad"))), (0.50, True)),
    cvs=[CV_weather],
)
I_U_excursionists_beach = Index(
    "excursionist beach usage factor",
    Piecewise((0.35, Eq(CV_weather.node, Symbol("bad"))), (0.80, True)),
    cvs=[CV_weather],
)

I_U_tourists_accommodation = Index("tourist accommodation usage factor", 0.90)

I_U_tourists_food = Index("tourist food service usage factor", 0.20)
I_U_excursionists_food = Index(
    "excursionist food service usage factor",
    Piecewise((0.80, Eq(CV_weather.node, Symbol("bad"))), (0.40, True)),
    cvs=[CV_weather, CV_weekday],
)

# Conversion indexes

I_Xa_tourists_per_vehicle = Index("tourists per vehicle allocation factor", 2.5)
I_Xa_excursionists_per_vehicle = Index("excursionists per vehicle allocation factor", 2.5)
I_Xo_tourists_parking = Index("tourists in parking rotation factor", 1.02)
I_Xo_excursionists_parking = Index("excursionists in parking rotation factor", 3.5)

I_Xo_tourists_beach = UniformDistIndex(
    "tourists on beach rotation factor",
    loc=1.0,
    scale=2.0,
)
I_Xo_excursionists_beach = Index("excursionists on beach rotation factor", 1.02)

I_Xa_tourists_accommodation = Index("tourists per accommodation allocation factor", 1.05)

I_Xa_visitors_food = Index("visitors in food service allocation factor", 0.9)
I_Xo_visitors_food = Index("visitors in food service rotation factor", 2.0)

# Presence indexes

I_P_tourists_reduction_factor = Index("tourists reduction factor", 1.0)
I_P_excursionists_reduction_factor = Index("excursionists reduction factor", 1.0)

I_P_tourists_saturation_level = Index("tourists saturation level", 10000)
I_P_excursionists_saturation_level = Index("excursionists saturation level", 10000)


# Constraints

C_parking = Constraint(
    usage=PV_tourists.node * I_U_tourists_parking.node / (I_Xa_tourists_per_vehicle.node * I_Xo_tourists_parking.node)
    + PV_excursionists.node
    * I_U_excursionists_parking.node
    / (I_Xa_excursionists_per_vehicle.node * I_Xo_excursionists_parking.node),
    capacity=I_C_parking.node,
    name="C_parking",
)

C_beach = Constraint(
    usage=PV_tourists.node * I_U_tourists_beach.node / I_Xo_tourists_beach.node
    + PV_excursionists.node * I_U_excursionists_beach.node / I_Xo_excursionists_beach.node,
    capacity=I_C_beach.node,
    name="C_beach",
)

# TODO: also capacity should be a formula
# C_accommodation = Constraint(usage=PV_tourists * I_U_tourists_accommodation,
#                              capacity=I_C_accommodation *  I_Xa_tourists_accommodation)

C_accommodation = Constraint(
    usage=PV_tourists.node * I_U_tourists_accommodation.node / I_Xa_tourists_accommodation.node,
    capacity=I_C_accommodation.node,
    name="C_accommodation",
)

# TODO: also capacity should be a formula
# C_food = Constraint(usage=PV_tourists * I_U_tourists_food +
#                              PV_excursionists * I_U_excursionists_food,
#                     capacity=I_C_food * I_Xa_visitors_food * I_Xo_visitors_food)
C_food = Constraint(
    usage=(PV_tourists.node * I_U_tourists_food.node + PV_excursionists.node * I_U_excursionists_food.node)
    / (I_Xa_visitors_food.node * I_Xo_visitors_food.node),
    capacity=I_C_food.node,
    name="C_food",
)


# Models
# TODO: what is the better process to create a model? (e.g., adding elements incrementally)

# Base model
M_Base = Model(
    "base model",
    [CV_weekday, CV_season, CV_weather],
    [PV_tourists, PV_excursionists],
    [
        I_U_tourists_parking,
        I_U_excursionists_parking,
        I_U_tourists_beach,
        I_U_excursionists_beach,
        I_U_tourists_accommodation,
        I_U_tourists_food,
        I_U_excursionists_food,
        I_Xa_tourists_per_vehicle,
        I_Xa_excursionists_per_vehicle,
        I_Xa_tourists_accommodation,
        I_Xo_tourists_parking,
        I_Xo_excursionists_parking,
        I_Xo_tourists_beach,
        I_Xo_excursionists_beach,
        I_Xa_visitors_food,
        I_Xo_visitors_food,
        I_P_tourists_reduction_factor,
        I_P_excursionists_reduction_factor,
        I_P_tourists_saturation_level,
        I_P_excursionists_saturation_level,
    ],
    [I_C_parking, I_C_beach, I_C_accommodation, I_C_food],
    [C_parking, C_beach, C_accommodation, C_food],
)

# Larger park capacity model
I_C_parking_larger = UniformDistIndex(
    "larger parking capacity",
    loc=550.0,
    scale=100.0,
)

# TODO(bassosimone): this is not working yet
try:
    M_MoreParking = M_Base.variation("larger parking model", change_capacities={I_C_parking: I_C_parking_larger})
except Exception:
    pass

# ANALYSIS SITUATIONS

# Base situation
S_Base = {}

# Good weather situation
S_Good_Weather = {CV_weather: [Symbol("good")]}

# Bad weather situation
S_Bad_Weather = {CV_weather: [Symbol("bad")]}

# PLOTTING

(t_max, e_max) = (10000, 10000)
(t_sample, e_sample) = (100, 100)
target_presence_samples = 200
ensemble_size = 20  # TODO: make configurable; may it be a CV parameter?
