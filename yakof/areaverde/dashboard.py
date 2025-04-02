from datetime import datetime, timedelta
import streamlit as st
from streamlit_option_menu import option_menu

from areaverde_yak import *


def to_time(seconds):
    return pd.Timestamp('00:00:00').to_pydatetime() + pd.Timedelta(seconds=seconds)


def to_number(time):
    return (time - pd.Timestamp('00:00:00')).total_seconds()

costs = [
             {'id': I_P_cost[e], 'label': f"Costo ingresso Euro {e}", 'value': I_P_cost[e].value,
              'min': 0.0, 'max': 10.0, 'step': 0.25} for e in range(7)
         ]

params = [
             {'id': I_P_start_time, 'label': "Ora inizio", 'value': to_time(I_P_start_time.value),
              'min': to_time(0), 'max': to_time(3600 * 24 - 900), 'step': timedelta(seconds=900), 'type': 'Time'},
             {'id': I_P_end_time, 'label': "Ora termine", 'value': to_time(I_P_end_time.value),
              'min': to_time(0), 'max': to_time(3600 * 24 - 900), 'step': timedelta(seconds=900), 'type': 'Time'},
             {'id': I_P_fraction_exempted, 'label': "% veicoli esonerati", 'value': I_P_fraction_exempted.value,
              'min': 0.0, 'max': 1.0, 'step': 0.05},
         ]

behaviors = [
    {'id': I_B_p50_cost, 'label': "Soglia di costo accettabile (mediano)", 'value': (I_B_p50_cost, I_B_p50_cost),
     'min': 0.0, 'max': 10.0, 'step': 0.25},
    {'id': I_B_p50_anticipating, 'label': "Disponibilità anticipo viaggio (in ore, mediano)", 'value': I_B_p50_anticipating.value,
     'min': 0.0, 'max': 6.0, 'step': 0.10},
    {'id': I_B_p50_postponing, 'label': "Disponibilità posticipo viaggio (in ore, mediano)", 'value': I_B_p50_postponing.value,
     'min': 0.0, 'max': 6.0, 'step': 0.10},
    {'id': I_B_p50_anticipation, 'label': "Tempo mediano di antico (in ore)", 'value': I_B_p50_anticipation.value,
     'min': 0.0, 'max': 6.0, 'step': 0.10},
    {'id': I_B_p50_postponement, 'label': "Tempo mediano mediano postico (in ore)", 'value': I_B_p50_postponement.value,
     'min': 0.0, 'max': 6.0, 'step': 0.10},
    {'id': I_B_starting_modified_factor, 'label': "Modifica circolazione in Area Verde", 'value': I_B_starting_modified_factor.value,
     'min': 0.0, 'max': 2.0, 'step': 0.10},
]


#################################################################################
#####################          STREAMLIT APP              #######################
#################################################################################
st.set_page_config(
    page_title="Area Verde",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    selected = option_menu(
        menu_title="Area Verde",
        options=["Home"],
        icons=["house"],
        menu_icon="cast",
        default_index=0,
    )

if selected == "Home":
    c = {}

    # Create a row layout
    c_plot, c_form = st.columns([2, 1])

    # Parameters
    with c_form:
        c_form.header("Parametri")
        with st.expander("Costi", False):
            for p in costs:
                if p['id'] not in st.session_state: st.session_state[p['id']] = p['value']
                format = 'HH:mm' if 'type' in p.keys() and p['type'] == 'Time' else None
                c[p['id']] = st.slider(p['label'], min_value=p['min'], max_value=p['max'], step=p['step'], format=format, key=p['id'])
        with st.expander("Parametri regolamento", False):
            for p in params:
                if p['id'] not in st.session_state: st.session_state[p['id']] = p['value']
                format = 'HH:mm' if 'type' in p.keys() and p['type'] == 'Time' else None
                c[p['id']] = st.slider(p['label'], min_value=p['min'], max_value=p['max'], step=p['step'], format=format, key=p['id'])
        with st.expander("Parametri comportamentali", False):
            for p in behaviors:
                if p['id'] not in st.session_state: st.session_state[p['id']] = p['value']
                format = 'HH:mm' if 'type' in p.keys() and p['type'] == 'Time' else None
                c[p['id']] = st.slider(p['label'], min_value=p['min'], max_value=p['max'], step=p['step'], format=format, key=p['id'])

    with c_plot:
        for p in costs + params + behaviors:
            if 'type' in p.keys() and p['type'] == 'Time':
                if isinstance(p['id'], UniformDistIndex):
                    (l,s) = st.session_state[p['id']]
                    (p['id'].loc, p['id'].scale) = (to_number(l), to_number(s))
                else:
                    p['id'].value = to_number(st.session_state[p['id']])
            else:
                if isinstance(p['id'], UniformDistIndex):
                    (p['id'].loc, p['id'].scale) = st.session_state[p['id']]
                else:
                    p['id'].value = st.session_state[p['id']]

        subs = evaluate(20)

        st.subheader("Veicoli in ingresso")
        plot_field_graph(subs[I_reduced_flow],
                         horizontal_label="Ora", vertical_label="Flusso (veicoli/ora)",
                         vertical_size=1250,
                         vertical_formatter=FuncFormatter(lambda x, _: f"{int(x * 12)}"),
                         reference_line=subs[TS_inflow][0])
        st.pyplot(plt, use_container_width=False, clear_figure=True)

        st.subheader("Traffico in Area Verde")
        plot_field_graph(subs[I_reduced_traffic],
                         horizontal_label="Ora", vertical_label="Traffico (veicoli circolanti)",
                         vertical_size=15000,
                         reference_line=subs[I_traffic][0])
        st.pyplot(plt, use_container_width=False, clear_figure=True)

        st.subheader("Emissioni in Area Verde")
        plot_field_graph(subs[I_reduced_emissions],
                         horizontal_label="Time", vertical_label="Emissions (NOx g/h)",
                         vertical_size=3000,
                         vertical_formatter=FuncFormatter(lambda x, _: f"{int(x * 12)}"),
                         reference_line=subs[I_emissions][0])
        st.pyplot(plt, use_container_width=False, clear_figure=True)

        kpi_translation = {
            'Base flow': 'Flusso attuale',
            'Reduced flow': 'Flusso ridotto',
            'Shifted flow': 'Flusso anticipato/posticipato',
            'Paying flow': 'Veicoli paganti',
            'Collected fees': 'Pagamenti collezionati',
            'Reduced emissions (NOx gr/day)': 'Riduzione emissioni (NOx gr/giorno)'
        }

        st.subheader("Indicatori (giornalieri)")
        for k,v in compute_kpis(subs).items():
            if k in kpi_translation:
                k = kpi_translation[k]
            st.write(f'{k} - {v:_}'.replace('_','.'))