import numpy as np
import pandas as pd

from yakof.sympyke import Piecewise
from yakof.frontend import graph
from yakof.dtyak import Index, UniformDistIndex


euro_class_split = [Index('euro_0',0.059),
                    Index('euro_1',0.012),
                    Index('euro_2',0.034),
                    Index('euro_3',0.054),
                    Index('euro_4',0.198),
                    Index('euro_5',0.176),
                    Index('euro_6',0.467)]

# Emissions nox per car per km
euro_class_emission = [Index('euro_0', 0.210584391986267347),
                    Index('euro_1', 0.2174573179869368),
                    Index('euro_2', 0.24014520073869067),
                    Index('euro_3', 0.24723923486567853),
                    Index('euro_4', 0.1355550834386541),
                    Index('euro_5', 0.09955851060544411),
                    Index('euro_6', 0.06824599009858062)]

# NOTE:‌ value for these index will set in the evaluation state
# TODO(maryam): this is just a hack but we need to figure out if we need indexes w/o a value
TS = Index('time range', 0)
TS_inflow = Index('inflow', 0) 
TS_starting = Index('starting', 0) 

I_P_start_time = Index('start time', (pd.Timestamp('07:30:00') - pd.Timestamp('00:00:00')).total_seconds())
I_P_end_time = Index('end time', (pd.Timestamp('19:30:00') - pd.Timestamp('00:00:00')).total_seconds())

I_P_cost = [Index(f'cost_euro_{e}', 5.00-e*0.25) for e in range(7)]
I_P_fraction_exempted = Index('exempted vehicles %', 0.15)
I_B_p50_cost = UniformDistIndex('cost 50% threshold', loc=4.00, scale=7.00) 
I_B_p50_anticipating = Index('anticipation 50% likelihood', 0.5)
I_B_p50_anticipation = Index('anticipation distribution 50% threshold', 0.25) 
I_B_p50_postponing = Index('postponement 50% likelihood', 0.8)
I_B_p50_postponement = Index('postponement distribution 50% threshold', 0.50)
I_B_starting_modified_factor = Index('starting modified factor', 1.00)

I_avg_cost = Index('average cost', sum(I_P_cost[i].node * euro_class_split[i].node for i in range(7)))
         
# TODO:make graph.log to accept scalar as an input.
I_fraction_rigid_euro = [Index(f'rigid vehicles euro_{e} %',
                         (1 - I_P_fraction_exempted.node) * graph.exp(I_P_cost[e].node / I_B_p50_cost.node * graph.log(graph.constant(0.5))),
                             ) for e in range(7) ]

I_fraction_rigid = Index('rigid vehicles %', 
                        sum(I_fraction_rigid_euro[i].node * euro_class_split[i].node for i in range(7))
                    )

I_reduced_euro_class_split = [Index(f'reduced split euro_{e} %',
                                     euro_class_split[e].node * (I_P_fraction_exempted.node + I_fraction_rigid_euro[e].node) /
                                     (I_P_fraction_exempted.node + I_fraction_rigid.node)) for e in range(7) ]

I_delta_from_start = Index('delta time from start',
                           Piecewise(((TS.node - I_P_start_time.node) / pd.Timedelta('1h').total_seconds(), TS.node >= I_P_start_time.node),
                                     (np.inf, True)))

I_fraction_anticipating = Index('anticipating vehicles %',
                                graph.exp(I_delta_from_start.node / I_B_p50_anticipating.node * graph.log(graph.constant(0.5))) *
                                (1 - I_P_fraction_exempted.node - I_fraction_rigid.node))

I_number_anticipating = Index('anticipating vehicles', I_fraction_anticipating.node * TS_inflow.node)

I_delta_to_end = Index('delta time to end',
                       Piecewise(((I_P_end_time.node - TS.node) / pd.Timedelta('1h').total_seconds(), TS.node <= I_P_end_time.node),
                                 (np.inf, True)))

I_fraction_postponing = Index('postponing vehicles %',
                              graph.exp(I_delta_to_end.node / I_B_p50_postponing.node * graph.log(graph.constant(0.5))) *
                              (1 - I_P_fraction_exempted.node - I_fraction_rigid.node))

I_number_postponing = Index('postponing vehicles', I_fraction_postponing.node * TS_inflow.node)

I_total_anticipating = Index('total anticipating vehicles', I_number_anticipating.node)

I_total_postponing = Index('total postponing vehicles', I_number_postponing.node)

I_delta_before_start = Index('delta time before start',
                             Piecewise(
                                 ((I_P_start_time.node - TS.node) / pd.Timedelta('1h').total_seconds(), TS.node < I_P_start_time.node),
                                 (np.inf, True)))

I_number_anticipated = Index('anticipated vehicles',
                             graph.exp(I_delta_before_start.node / I_B_p50_anticipation.node * graph.log(
                                 graph.constant(0.5))) / I_B_p50_anticipation.node * graph.log(graph.constant(2)) / 12 * I_total_anticipating.node)

I_delta_after_end = Index('delta time after end',
                          Piecewise(((TS.node - I_P_end_time.node) / pd.Timedelta('1h').total_seconds(), TS.node > I_P_end_time.node),
                                    (np.inf, True)))

I_number_postponed = Index('postponed vehicles',
                           graph.exp(I_delta_after_end.node / I_B_p50_postponement.node * graph.log(graph.constant(0.5))) / I_B_p50_postponement.node * 
                           graph.log(graph.constant(2)) / 12 * I_total_postponing.node)


I_number_shifted = Index('shifted vehicles', I_number_anticipated.node + I_number_postponed.node)

I_reduced_flow = Index('reduced vehicle flow',
                       Piecewise(((I_P_fraction_exempted.node + I_fraction_rigid.node) * TS_inflow.node,
                                  (TS.node >= I_P_start_time.node) & (TS.node <= I_P_end_time.node)),
                                 (TS_inflow.node + I_number_shifted.node, True)))

I_total_base_flow = Index('total base vehicle flow', TS_inflow.node)

I_total_reduced_flow = Index('total reduced vehicle flow', I_reduced_flow.node)

I_number_paying = Index('paying vehicles',
                        Piecewise((I_fraction_rigid.node * TS_inflow.node,
                                   (TS.node >= I_P_start_time.node) & (TS.node <= I_P_end_time.node)),
                                  (0, True)))

I_total_paying = Index('total vehicles paying', I_number_paying.node)


I_total_payed = Index('total payed fees', I_total_paying.node * I_avg_cost.node)

I_total_anticipated = Index('total vehicles anticipated', I_number_anticipated.node)

I_total_postponed = Index('total vehicles postponed', I_number_postponed.node)
                
I_total_shifted = Index('total vehicles shifted', I_total_anticipated.node + I_total_postponed.node)

I_modified_starting = Index('modified starting', TS_starting.node + I_reduced_flow.node * (I_B_starting_modified_factor.node - 1))

# NOTE: I_traffic originally was TS_solve(TS_inflow.node + TS_starting.node)
I_traffic = Index('reference traffic', TS_inflow.node + TS_starting.node)

I_reduced_traffic = Index('modified traffic', I_reduced_flow.node + I_modified_starting.node)

I_average_emissions = Index('average emissions (per vehicle, per km)', 
                            euro_class_emission[0].node * euro_class_split[0].node +
                            euro_class_emission[1].node * euro_class_split[1].node +
                            euro_class_emission[2].node * euro_class_split[2].node +
                            euro_class_emission[3].node * euro_class_split[3].node +
                            euro_class_emission[4].node * euro_class_split[4].node +
                            euro_class_emission[5].node * euro_class_split[5].node +
                            euro_class_emission[6].node * euro_class_split[6].node
                        )
    

I_reduced_average_emissions = Index('reduced average emissions (per vehicle, per km)',
                                    sum(euro_class_emission[i].node * I_reduced_euro_class_split[i].node for i in range(7)))

# TODO: improve - at the moment, the conversion factor is 2,5 km per 5 minutes
# here I_traffic is not calculated and I_emission is not lazy calculation   
I_emissions = Index('emissions', 2.5 * I_average_emissions.node * I_traffic.node)



# TODO: The average emissions is probably different outside regulated hours
#  (shifted cars' emissions are probably proportional to shifted cars' euro level mix)  
I_reduced_emissions = Index('reduced emissions',
                            Piecewise((2.5 * I_reduced_average_emissions.node * I_reduced_traffic.node,
                                       (TS.node >= I_P_start_time.node) & (TS.node <= I_P_end_time.node)),
                                      (2.5 * I_average_emissions.node * I_reduced_traffic.node, True)))

I_total_emissions = Index('total emissions', I_emissions.node)

I_total_reduced_emissions = Index('total reduced emissions', I_reduced_emissions.node)

indexes = [TS, TS_inflow, TS_starting,
           I_P_start_time, I_P_end_time, *I_P_cost, I_P_fraction_exempted,
           I_B_p50_cost, I_B_p50_anticipating, I_B_p50_anticipation, I_B_p50_postponing, I_B_p50_postponement,
           I_B_starting_modified_factor,
           I_avg_cost, *I_fraction_rigid_euro, I_fraction_rigid, *I_reduced_euro_class_split,
           I_delta_from_start, I_fraction_anticipating, I_number_anticipating,
           I_delta_to_end, I_fraction_postponing, I_number_postponing,
           I_total_anticipating, I_total_postponing, I_delta_before_start, I_number_anticipated,
           I_delta_after_end, I_number_postponed, I_number_shifted,
           I_reduced_flow, I_total_base_flow, I_total_reduced_flow,
           I_number_paying, I_total_paying, I_total_payed,
           I_total_anticipated, I_total_postponed, I_total_shifted,
           I_modified_starting, I_traffic, I_reduced_traffic,
           I_average_emissions, I_reduced_average_emissions,
           I_emissions, I_reduced_emissions, I_total_emissions, I_total_reduced_emissions ]


# def evaluate(size=1):
#     subs = {}
#     for index in indexes:
#         if index.cvs is None:
#             if isinstance(index.value, numbers.Number):
#                 subs[index] = np.expand_dims(np.array([index.value] * size), axis=1)
#             elif isinstance(index.value, np.ndarray):
#                 subs[index] = np.expand_dims(index.value, axis=0)
#             else:
#                 subs[index] = np.expand_dims(np.array(index.value.rvs(size=size)), axis=1)
#         else:
#             args = [subs[cv] for cv in index.cvs]
#             subs[index] = index.value(*args)
#     return subs


# def distribution(field, size=10000, num=100):
#     xx, yy = np.meshgrid(np.linspace(0, size, num + 1), range(field.shape[1]))
#     zz = stats.poisson(mu=np.expand_dims(field, axis=2)).cdf(np.expand_dims(xx, axis=0))
#     return zz.mean(axis=0)


# def plot_field_graph(field, horizontal_label, vertical_label, vertical_size, vertical_formatter=None,
#                      reference_line=None):
#     dist = distribution(field, vertical_size, 100)
#     plt.figure(figsize=(8, 6))
#     plt.pcolormesh(pd.date_range(start='00:00:00', periods=12 * 24, freq='5min'),
#                    np.linspace(0, vertical_size, 100 + 1), dist.T,
#                    cmap='coolwarm_r', vmin=0.0, vmax=1.0)
#     if reference_line is not None:
#         plt.plot(pd.date_range(start='00:00:00', periods=12 * 24, freq='5min'),
#                  reference_line, linewidth=2, color='black')
#     plt.plot(pd.date_range(start='00:00:00', periods=12 * 24, freq='5min'),
#              field.mean(axis=0), linewidth=1, color='black')
#     plt.gca().set_ylim([0, vertical_size])
#     if vertical_formatter is not None:
#         plt.gca().yaxis.set_major_formatter(vertical_formatter)
#     plt.gca().set_ylabel(vertical_label)
#     plt.gcf().tight_layout()
#     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
#     plt.gcf().autofmt_xdate()
#     plt.gca().set_xlabel(horizontal_label)


# def compute_kpis(evals):
#     return {
#         'Base flow': int(evals[I_total_base_flow].mean()),
#         'Reduced flow': int(evals[I_total_reduced_flow].mean()),
#         'Shifted flow': int(evals[I_total_shifted].mean()),
#         'Paying flow': int(evals[I_total_paying].mean()) if evals[I_avg_cost].mean() > 0 else 0,
#         'Collected fees': int(evals[I_total_payed].mean()),
#         'Reduced emissions (NOx gr/day)': int(evals[I_total_emissions].mean()) - int(evals[I_total_reduced_emissions].mean())
#     }


# if __name__ == "__main__":
#     subs = evaluate(20)

#     plot_field_graph(subs[I_reduced_flow],
#                      horizontal_label="Time", vertical_label="Flow (vehicles/hour)",
#                      vertical_size=1250,
#                      vertical_formatter=FuncFormatter(lambda x, _: f"{int(x * 12)}"),
#                      reference_line=subs[TS_inflow][0])
#     plt.show()

#     plot_field_graph(subs[I_reduced_traffic],
#                      horizontal_label="Time", vertical_label="Traffic (circulating vehicles)",
#                      vertical_size=15000,
#                      reference_line=subs[I_traffic][0])
#     plt.show()

#     plot_field_graph(subs[I_reduced_emissions],
#                      horizontal_label="Time", vertical_label="Emissions (NOx g/h)",
#                      vertical_size=3000,
#                      vertical_formatter=FuncFormatter(lambda x, _: f"{int(x * 12)}"),
#                      reference_line=subs[I_emissions][0])
#     plt.show()

#     for k, v in compute_kpis(subs).items():
#         print(f'{k} - {v:,}')