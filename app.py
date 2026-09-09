import copy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from physics import (
    SimParams,
    run_simulation,
    decision_support,
    extra_time_from_cooling_h,
    extra_time_from_withdrawal_h,
    RHO_LIQUID_H2,
    H_FG_H2,
    T_NBP,
)

st.set_page_config(
    page_title="LH2 Storage Tank Digital Twin",
    page_icon="🧊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(128,128,128,0.22);
        padding: 14px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧊 Liquid Hydrogen (LH₂) Storage Tank — Digital Twin")
st.caption(
    "Engineering prototype / decision-support tool — NOT a certified safety model. "
    "Predicts boil-off, pressure rise, time-to-venting, and compares mitigation strategies."
)

with st.expander("Engineering assumptions used in this model"):
    st.markdown(
        f"""
- **0-D lumped transient model** with bulk liquid and vapor/ullage phases.
- Net heat ingress creates boil-off using a constant latent heat:
  **h_fg ≈ {H_FG_H2/1000:.1f} kJ/kg**.
- Liquid hydrogen density is approximated as
  **ρ ≈ {RHO_LIQUID_H2:.1f} kg/m³**.
- Vapor pressure is estimated with an ideal-gas ullage model.
- Tank temperature is approximated from a simple saturation relation.
- Venting starts when the specified pressure limit is reached.
- Withdrawal removes liquid and increases ullage volume.
- Heat ingress can be represented by **UA·ΔT** or by a fixed heat leak.
- Internal calculations use SI units.

This model is for demonstration and competition/prototype work only.
        """
    )

# ---------------- SIDEBAR ----------------
st.sidebar.header("Tank & Initial Conditions")
V_tank = st.sidebar.number_input(
    "Tank total volume (m³)", min_value=0.1, max_value=100000.0,
    value=50.0, step=1.0
)
fill_level = st.sidebar.slider(
    "Initial LH₂ fill level (%)", 1.0, 99.0, 80.0, step=1.0
)
P_init = st.sidebar.number_input(
    "Initial tank pressure (bar abs)", min_value=0.05, max_value=13.0,
    value=1.5, step=0.05
)
T_init = st.sidebar.slider(
    "Initial LH₂ temperature (K)", 14.0, 33.0, 21.0, step=0.1
)
T_amb_C = st.sidebar.slider(
    "Ambient temperature (°C)", -50.0, 60.0, 20.0, step=1.0
)

st.sidebar.header("Storage Duration")
duration_unit = st.sidebar.radio(
    "Duration unit", ["hours", "days"], horizontal=True
)
duration_val = st.sidebar.number_input(
    f"Storage duration ({duration_unit})",
    min_value=0.1,
    value=72.0 if duration_unit == "hours" else 3.0,
    step=1.0,
)
duration_h = duration_val if duration_unit == "hours" else duration_val * 24.0

st.sidebar.header("Heat Leak Model")
heat_mode_label = st.sidebar.radio(
    "Heat input mode", ["UA (W/K)", "Fixed heat leak (W)"]
)
heat_input_mode = "UA" if heat_mode_label.startswith("UA") else "FIXED_LEAK"

if heat_input_mode == "UA":
    UA = st.sidebar.number_input(
        "Overall UA (W/K)", min_value=0.0, max_value=10000.0,
        value=5.0, step=0.5
    )
    Q_leak_fixed = 0.0
else:
    Q_leak_fixed = st.sidebar.number_input(
        "Fixed total heat leak (W)", min_value=0.0, max_value=1_000_000.0,
        value=500.0, step=10.0
    )
    UA = 0.0

st.sidebar.header("Venting / Safety")
P_vent = st.sidebar.number_input(
    "Venting pressure limit (bar abs)", min_value=0.2, max_value=15.0,
    value=5.0, step=0.1
)
safety_margin = st.sidebar.slider(
    "Safety margin below vent limit (bar)", 0.0, 3.0, 0.3, step=0.1
)

st.sidebar.header("Mitigation")
cooling_W = st.sidebar.number_input(
    "Active cooling duty (W)", min_value=0.0, max_value=100000.0,
    value=0.0, step=50.0
)
cooling_start_h = st.sidebar.number_input(
    "Cooling start time (h)", min_value=0.0, value=0.0, step=1.0
)
withdrawal_kgph = st.sidebar.number_input(
    "Hydrogen withdrawal rate (kg/h)", min_value=0.0, max_value=100000.0,
    value=0.0, step=1.0
)
withdrawal_start_h = st.sidebar.number_input(
    "Withdrawal start time (h)", min_value=0.0, value=0.0, step=1.0
)

st.sidebar.header("Operating Mode")
mode_label = st.sidebar.selectbox(
    "Mode to simulate",
    [
        "A - No intervention",
        "B - Predictive active cooling",
        "C - Predictive hydrogen withdrawal",
    ],
)
mode = mode_label[0]

with st.sidebar.expander("Advanced"):
    dt_s = st.number_input(
        "Simulation time step (s)",
        min_value=1.0,
        max_value=3600.0,
        value=60.0,
        step=1.0,
    )

params = SimParams(
    V_tank_m3=V_tank,
    fill_level_pct=fill_level,
    P_init_bar=P_init,
    T_init_K=T_init,
    T_ambient_C=T_amb_C,
    duration_h=duration_h,
    heat_input_mode=heat_input_mode,
    UA_W_per_K=UA,
    Q_leak_fixed_W=Q_leak_fixed,
    P_vent_bar=P_vent,
    safety_margin_bar=safety_margin,
    cooling_W=cooling_W,
    cooling_start_h=cooling_start_h,
    withdrawal_kg_h=withdrawal_kgph,
    withdrawal_start_h=withdrawal_start_h,
    dt_s=dt_s,
)

try:
    params.validate()
except ValueError as e:
    st.error(str(e))
    st.stop()

result_selected = run_simulation(params, mode=mode)
result_A = run_simulation(params, mode="A")
ds = decision_support(params)

# ---------------- STATUS ----------------
status = ds["status"]
status_style = {
    "GREEN": ("🟢", "#1e7e34", "SAFE for the requested storage period"),
    "YELLOW": ("🟡", "#b8860b", "INTERVENTION RECOMMENDED"),
    "RED": ("🔴", "#c0392b", "VENTING PREDICTED — action required"),
}
icon, color, label = status_style[status]

st.markdown(
    f"""
    <div style="padding:14px 20px;border-radius:12px;
    background-color:{color}18;border:2px solid {color};margin-bottom:12px;">
      <span style="font-size:26px;">{icon}</span>
      <span style="font-size:22px;font-weight:700;color:{color};"> {label}</span>
      <div style="margin-top:6px;">{ds['message']}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- METRICS ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current tank pressure", f"{result_selected['P_bar'][0]:.2f} bar(a)")
c2.metric(
    "Predicted max pressure",
    f"{result_selected['max_pressure_bar']:.2f} bar(a)",
    delta=f"Vent limit {P_vent:.2f} bar(a)",
    delta_color="off",
)
ttv = result_selected["time_to_vent_h"]
c3.metric("Time to vent", f"{ttv:.2f} h" if ttv is not None else "No venting predicted")
c4.metric(
    "Initial boil-off rate",
    f"{result_selected['boiloff_rate_kg_h_initial']:.2f} kg/h",
    f"{result_selected['boiloff_pct_per_day_initial']:.3f} %/day",
)

c5, c6, c7, c8 = st.columns(4)
c5.metric("Cumulative BOG", f"{result_selected['total_bog_kg']:.1f} kg")
c6.metric("Cumulative withdrawn", f"{result_selected['total_withdrawn_kg']:.1f} kg")
c7.metric("Cumulative vented", f"{result_selected['total_vented_kg']:.1f} kg")
c8.metric("Recommended action", ds["recommended_action"])

req_cool = ds.get("required_cooling_W")
req_with = ds.get("required_withdrawal_kgph")
c9, c10 = st.columns(2)
c9.metric(
    "Required cooling duty to avoid venting",
    f"{req_cool:.0f} W" if req_cool is not None else "—",
)
c10.metric(
    "Required H₂ withdrawal to avoid venting",
    f"{req_with:.1f} kg/h" if req_with is not None else "—",
)

if status == "YELLOW":
    colx, coly = st.columns(2)
    extra_cool_h = extra_time_from_cooling_h(params, req_cool) if req_cool else None
    extra_with_h = extra_time_from_withdrawal_h(params, req_with) if req_with else None
    with colx:
        st.info(
            f"**Extra safe storage time from cooling:** "
            f"{extra_cool_h:.2f} h" if extra_cool_h is not None else "N/A"
        )
    with coly:
        st.info(
            f"**Extra safe storage time from withdrawal:** "
            f"{extra_with_h:.2f} h" if extra_with_h is not None else "N/A"
        )

st.divider()

# ---------------- PLOTS ----------------
st.subheader("Simulation results")
t_h = result_selected["t_h"]

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Pressure vs Time",
    "Boil-off Rate vs Time",
    "LH₂ Inventory vs Time",
    "Heat Ingress vs Time",
    "Mode Comparison",
])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_h, y=result_selected["P_bar"], mode="lines", name=f"Mode {mode}"))
    fig.add_hline(y=P_vent, line_dash="dash", annotation_text="Vent limit")
    fig.add_hline(y=P_vent - safety_margin, line_dash="dot", annotation_text="Safety margin")
    fig.update_layout(xaxis_title="Time (h)", yaxis_title="Pressure (bar abs)", height=420)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_h, y=result_selected["mdot_bog_kg_h"], mode="lines", name="Boil-off rate"))
    fig.update_layout(xaxis_title="Time (h)", yaxis_title="Boil-off rate (kg/h)", height=420)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_h, y=result_selected["m_liquid_kg"], mode="lines", name="LH₂ liquid mass"))
    fig.add_trace(go.Scatter(x=t_h, y=result_selected["m_vapor_kg"], mode="lines", name="Vapor mass"))
    fig.update_layout(xaxis_title="Time (h)", yaxis_title="Mass (kg)", height=420)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_h, y=result_selected["Q_in_W"], mode="lines", name="Heat ingress"))
    fig.add_trace(go.Scatter(x=t_h, y=result_selected["Q_cool_W"], mode="lines", name="Cooling"))
    fig.update_layout(xaxis_title="Time (h)", yaxis_title="Heat rate (W)", height=420)
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    result_B = run_simulation(params, mode="B")
    result_C = run_simulation(params, mode="C")

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=result_A["t_h"], y=result_A["P_bar"], mode="lines", name="A: No intervention"))
    fig.add_trace(go.Scatter(x=result_B["t_h"], y=result_B["P_bar"], mode="lines", name=f"B: Cooling {cooling_W:.0f} W"))
    fig.add_trace(go.Scatter(x=result_C["t_h"], y=result_C["P_bar"], mode="lines", name=f"C: Withdrawal {withdrawal_kgph:.1f} kg/h"))
    fig.add_hline(y=P_vent, line_dash="dash", annotation_text="Vent limit")
    fig.update_layout(xaxis_title="Time (h)", yaxis_title="Pressure (bar abs)", height=460)
    st.plotly_chart(fig, use_container_width=True)

    comp_df = pd.DataFrame({
        "Mode": ["A - No intervention", "B - Active cooling", "C - Hydrogen withdrawal"],
        "Max pressure (bar)": [
            result_A["max_pressure_bar"],
            result_B["max_pressure_bar"],
            result_C["max_pressure_bar"],
        ],
        "Time to vent (h)": [
            f"{result_A['time_to_vent_h']:.2f}" if result_A["time_to_vent_h"] is not None else "No vent",
            f"{result_B['time_to_vent_h']:.2f}" if result_B["time_to_vent_h"] is not None else "No vent",
            f"{result_C['time_to_vent_h']:.2f}" if result_C["time_to_vent_h"] is not None else "No vent",
        ],
        "Total H₂ vented (kg)": [
            result_A["total_vented_kg"],
            result_B["total_vented_kg"],
            result_C["total_vented_kg"],
        ],
        "Final liquid mass (kg)": [
            result_A["final_liquid_kg"],
            result_B["final_liquid_kg"],
            result_C["final_liquid_kg"],
        ],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

st.divider()

# ---------------- SCENARIO COMPARISON ----------------
st.subheader("Scenario comparison")
st.caption(
    "Compare ambient temperature, insulation quality (UA), fill level, duration and active cooling."
)

default_scenarios = pd.DataFrame([
    {
        "Scenario": "Baseline",
        "Ambient T (°C)": T_amb_C,
        "UA (W/K)": UA,
        "Fill level (%)": fill_level,
        "Duration (h)": duration_h,
        "Cooling duty (W)": 0.0,
    },
    {
        "Scenario": "Hot day",
        "Ambient T (°C)": T_amb_C + 15,
        "UA (W/K)": UA,
        "Fill level (%)": fill_level,
        "Duration (h)": duration_h,
        "Cooling duty (W)": 0.0,
    },
    {
        "Scenario": "Degraded insulation",
        "Ambient T (°C)": T_amb_C,
        "UA (W/K)": UA * 2,
        "Fill level (%)": fill_level,
        "Duration (h)": duration_h,
        "Cooling duty (W)": 0.0,
    },
    {
        "Scenario": "With active cooling",
        "Ambient T (°C)": T_amb_C,
        "UA (W/K)": UA,
        "Fill level (%)": fill_level,
        "Duration (h)": duration_h,
        "Cooling duty (W)": ds.get("required_cooling_W") or 500.0,
    },
])

scenario_df = st.data_editor(
    default_scenarios,
    num_rows="dynamic",
    use_container_width=True,
    key="scenario_editor",
)

if st.button("▶ Run scenario comparison"):
    rows = []

    for _, row in scenario_df.iterrows():
        try:
            p_s = copy.deepcopy(params)
            p_s.T_ambient_C = float(row["Ambient T (°C)"])
            p_s.UA_W_per_K = float(row["UA (W/K)"])
            p_s.heat_input_mode = "UA"
            p_s.fill_level_pct = float(row["Fill level (%)"])
            p_s.duration_h = float(row["Duration (h)"])
            p_s.cooling_W = float(row["Cooling duty (W)"])
            p_s.cooling_start_h = 0.0
            p_s.validate()

            mode_s = "B" if p_s.cooling_W > 0 else "A"
            res_s = run_simulation(p_s, mode=mode_s)

            rows.append({
                "Scenario": row["Scenario"],
                "Max pressure (bar)": round(res_s["max_pressure_bar"], 2),
                "Time to vent (h)": round(res_s["time_to_vent_h"], 2)
                    if res_s["time_to_vent_h"] is not None else "No vent",
                "Initial boil-off (kg/h)": round(res_s["boiloff_rate_kg_h_initial"], 2),
                "Total vented (kg)": round(res_s["total_vented_kg"], 1),
                "Final liquid mass (kg)": round(res_s["final_liquid_kg"], 1),
            })

        except ValueError as e:
            rows.append({
                "Scenario": row.get("Scenario", "Unnamed"),
                "Max pressure (bar)": "Invalid inputs",
                "Time to vent (h)": str(e),
                "Initial boil-off (kg/h)": "-",
                "Total vented (kg)": "-",
                "Final liquid mass (kg)": "-",
            })

    results_df = pd.DataFrame(rows)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    numeric_rows = results_df[
        pd.to_numeric(results_df["Max pressure (bar)"], errors="coerce").notna()
    ]

    if not numeric_rows.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=numeric_rows["Scenario"],
            y=numeric_rows["Max pressure (bar)"],
        ))
        fig.add_hline(y=P_vent, line_dash="dash", annotation_text="Vent limit")
        fig.update_layout(
            yaxis_title="Max predicted pressure (bar abs)",
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption(
    "⚠️ Disclaimer: This is an engineering prototype / decision-support digital twin. "
    "It is not a certified safety, process design, or code-compliance model."
)
