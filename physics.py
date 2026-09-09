from dataclasses import dataclass
import copy
import math
import numpy as np

R_UNIVERSAL = 8.314462618  # J/mol/K
M_H2 = 2.01588e-3          # kg/mol
R_H2 = R_UNIVERSAL / M_H2  # J/kg/K

RHO_LIQUID_H2 = 70.8       # kg/m3, near normal boiling point
H_FG_H2 = 445_000.0        # J/kg, approximate latent heat near NBP
T_NBP = 20.27              # K at ~1.013 bar

# Approximate Clausius-Clapeyron constant using h_fg / R_specific
CC_B = H_FG_H2 / R_H2


def celsius_to_kelvin(T_C: float) -> float:
    return T_C + 273.15


def saturation_pressure_bar(T_K: float) -> float:
    """
    Simple Clausius-Clapeyron approximation anchored at the normal boiling point.
    Intended only for a prototype digital twin.
    """
    if T_K <= 0:
        raise ValueError("Temperature must be greater than 0 K.")
    P_ref_bar = 1.01325
    return P_ref_bar * math.exp(CC_B * (1.0 / T_NBP - 1.0 / T_K))


def saturation_temperature_K(P_bar: float) -> float:
    if P_bar <= 0:
        raise ValueError("Pressure must be positive.")
    P_ref_bar = 1.01325
    denom = (1.0 / T_NBP) - math.log(P_bar / P_ref_bar) / CC_B
    if denom <= 0:
        return 33.0
    return 1.0 / denom


def compressibility_factor(P_bar: float, T_K: float) -> float:
    # Hook for future replacement with a real-gas EOS.
    return 1.0


@dataclass
class SimParams:
    V_tank_m3: float
    fill_level_pct: float
    P_init_bar: float
    T_init_K: float
    T_ambient_C: float
    duration_h: float
    heat_input_mode: str
    UA_W_per_K: float
    Q_leak_fixed_W: float
    P_vent_bar: float
    safety_margin_bar: float
    cooling_W: float
    cooling_start_h: float
    withdrawal_kg_h: float
    withdrawal_start_h: float
    dt_s: float = 60.0

    def validate(self):
        if self.V_tank_m3 <= 0:
            raise ValueError("Tank volume must be positive.")
        if not (0 < self.fill_level_pct < 100):
            raise ValueError("Fill level must be between 0 and 100%.")
        if self.P_init_bar <= 0:
            raise ValueError("Initial pressure must be positive.")
        if self.P_vent_bar <= self.P_init_bar:
            raise ValueError("Venting pressure must be greater than initial pressure.")
        if self.duration_h <= 0:
            raise ValueError("Storage duration must be positive.")
        if self.dt_s <= 0:
            raise ValueError("Simulation time step must be positive.")
        if self.heat_input_mode not in ("UA", "FIXED_LEAK"):
            raise ValueError("Heat input mode must be 'UA' or 'FIXED_LEAK'.")
        if self.UA_W_per_K < 0 or self.Q_leak_fixed_W < 0:
            raise ValueError("Heat leak inputs cannot be negative.")
        if self.cooling_W < 0 or self.withdrawal_kg_h < 0:
            raise ValueError("Mitigation inputs cannot be negative.")
        if self.safety_margin_bar < 0:
            raise ValueError("Safety margin cannot be negative.")
        return True


def _initial_state(params: SimParams):
    V_liq = params.V_tank_m3 * params.fill_level_pct / 100.0
    m_liq = V_liq * RHO_LIQUID_H2
    V_ullage = max(params.V_tank_m3 - V_liq, 1e-6)

    P_pa = params.P_init_bar * 1e5
    Z = compressibility_factor(params.P_init_bar, params.T_init_K)
    m_vap = P_pa * V_ullage / (Z * R_H2 * params.T_init_K)

    return m_liq, m_vap


def run_simulation(params: SimParams, mode="A"):
    params.validate()
    if mode not in ("A", "B", "C"):
        raise ValueError("Mode must be A, B or C.")

    n_steps = max(2, int(math.ceil(params.duration_h * 3600.0 / params.dt_s)) + 1)
    t_s = np.linspace(0.0, params.duration_h * 3600.0, n_steps)

    P_bar = np.zeros(n_steps)
    T_K = np.zeros(n_steps)
    m_liq_arr = np.zeros(n_steps)
    m_vap_arr = np.zeros(n_steps)
    mdot_bog_kg_h = np.zeros(n_steps)
    Q_in_W = np.zeros(n_steps)
    Q_cool_W = np.zeros(n_steps)
    vent_rate_kg_h = np.zeros(n_steps)
    withdrawal_rate_kg_h = np.zeros(n_steps)

    m_liq, m_vap = _initial_state(params)
    P_bar[0] = params.P_init_bar
    T_K[0] = params.T_init_K
    m_liq_arr[0] = m_liq
    m_vap_arr[0] = m_vap

    total_bog = 0.0
    total_vented = 0.0
    total_withdrawn = 0.0
    time_to_vent_h = None

    for i in range(1, n_steps):
        dt = t_s[i] - t_s[i - 1]
        t_h_prev = t_s[i - 1] / 3600.0

        # Use previous pressure to estimate equilibrium tank temperature.
        T_tank = saturation_temperature_K(max(P_bar[i - 1], 0.05))
        T_K[i - 1] = T_tank

        # Heat ingress
        if params.heat_input_mode == "UA":
            q_in = max(0.0, params.UA_W_per_K * (celsius_to_kelvin(params.T_ambient_C) - T_tank))
        else:
            q_in = max(0.0, params.Q_leak_fixed_W)

        # Mitigation
        q_cool = 0.0
        if mode == "B" and t_h_prev >= params.cooling_start_h:
            q_cool = params.cooling_W

        q_net = max(0.0, q_in - q_cool)
        dm_bog = min(m_liq, q_net * dt / H_FG_H2)

        # Liquid withdrawal
        dm_with = 0.0
        if mode == "C" and t_h_prev >= params.withdrawal_start_h and m_liq > 0:
            dm_with = min(m_liq - dm_bog, max(0.0, params.withdrawal_kg_h) * dt / 3600.0)

        m_liq = max(0.0, m_liq - dm_bog - dm_with)
        m_vap = max(0.0, m_vap + dm_bog)

        total_bog += dm_bog
        total_withdrawn += dm_with

        # Updated ullage and pressure
        V_liq = m_liq / RHO_LIQUID_H2 if RHO_LIQUID_H2 > 0 else 0.0
        V_ullage = max(params.V_tank_m3 - V_liq, 1e-6)

        # Iterate a few times because temperature depends on pressure
        P_est_bar = max(P_bar[i - 1], 0.05)
        for _ in range(5):
            T_est = saturation_temperature_K(P_est_bar)
            Z = compressibility_factor(P_est_bar, T_est)
            P_new_bar = (m_vap * Z * R_H2 * T_est / V_ullage) / 1e5
            P_est_bar = max(P_new_bar, 0.01)

        # PRV behavior: vent enough vapor to hold at vent pressure
        if P_est_bar >= params.P_vent_bar:
            if time_to_vent_h is None:
                time_to_vent_h = t_s[i] / 3600.0

            T_vent = saturation_temperature_K(params.P_vent_bar)
            Z = compressibility_factor(params.P_vent_bar, T_vent)
            m_vap_allowed = params.P_vent_bar * 1e5 * V_ullage / (Z * R_H2 * T_vent)
            dm_vent = max(0.0, m_vap - m_vap_allowed)
            dm_vent = min(dm_vent, m_vap)
            m_vap -= dm_vent
            total_vented += dm_vent
            P_est_bar = params.P_vent_bar
            vent_rate_kg_h[i] = dm_vent / dt * 3600.0 if dt > 0 else 0.0

        P_bar[i] = P_est_bar
        T_K[i] = saturation_temperature_K(max(P_est_bar, 0.05))
        m_liq_arr[i] = m_liq
        m_vap_arr[i] = m_vap
        mdot_bog_kg_h[i] = dm_bog / dt * 3600.0 if dt > 0 else 0.0
        Q_in_W[i] = q_in
        Q_cool_W[i] = q_cool
        withdrawal_rate_kg_h[i] = dm_with / dt * 3600.0 if dt > 0 else 0.0

    # Fill initial point consistently
    T_K[0] = params.T_init_K
    if n_steps > 1:
        Q_in_W[0] = Q_in_W[1]
        Q_cool_W[0] = Q_cool_W[1]
        mdot_bog_kg_h[0] = mdot_bog_kg_h[1]
        withdrawal_rate_kg_h[0] = withdrawal_rate_kg_h[1]

    initial_liq, _ = _initial_state(params)
    initial_bor = mdot_bog_kg_h[0]
    bor_pct_day = (initial_bor * 24.0 / initial_liq * 100.0) if initial_liq > 0 else 0.0

    return {
        "t_h": t_s / 3600.0,
        "P_bar": P_bar,
        "T_K": T_K,
        "m_liquid_kg": m_liq_arr,
        "m_vapor_kg": m_vap_arr,
        "mdot_bog_kg_h": mdot_bog_kg_h,
        "Q_in_W": Q_in_W,
        "Q_cool_W": Q_cool_W,
        "vent_rate_kg_h": vent_rate_kg_h,
        "withdrawal_rate_kg_h": withdrawal_rate_kg_h,
        "max_pressure_bar": float(np.max(P_bar)),
        "time_to_vent_h": time_to_vent_h,
        "boiloff_rate_kg_h_initial": float(initial_bor),
        "boiloff_pct_per_day_initial": float(bor_pct_day),
        "total_bog_kg": float(total_bog),
        "total_withdrawn_kg": float(total_withdrawn),
        "total_vented_kg": float(total_vented),
        "final_liquid_kg": float(m_liq_arr[-1]),
    }


def _avoids_venting(params: SimParams, mode: str) -> bool:
    res = run_simulation(params, mode=mode)
    return res["time_to_vent_h"] is None


def find_min_cooling_duty_W(params: SimParams, max_search_W=200_000.0):
    baseline = run_simulation(params, mode="A")
    if baseline["time_to_vent_h"] is None:
        return 0.0

    low, high = 0.0, 100.0
    while high <= max_search_W:
        p = copy.deepcopy(params)
        p.cooling_W = high
        p.cooling_start_h = 0.0
        if _avoids_venting(p, "B"):
            break
        high *= 2.0
    else:
        return None

    for _ in range(35):
        mid = 0.5 * (low + high)
        p = copy.deepcopy(params)
        p.cooling_W = mid
        p.cooling_start_h = 0.0
        if _avoids_venting(p, "B"):
            high = mid
        else:
            low = mid

    return high


def find_min_withdrawal_kgph(params: SimParams, max_search_kgph=100_000.0):
    baseline = run_simulation(params, mode="A")
    if baseline["time_to_vent_h"] is None:
        return 0.0

    low, high = 0.0, 0.1
    while high <= max_search_kgph:
        p = copy.deepcopy(params)
        p.withdrawal_kg_h = high
        p.withdrawal_start_h = 0.0
        if _avoids_venting(p, "C"):
            break
        high *= 2.0
    else:
        return None

    for _ in range(35):
        mid = 0.5 * (low + high)
        p = copy.deepcopy(params)
        p.withdrawal_kg_h = mid
        p.withdrawal_start_h = 0.0
        if _avoids_venting(p, "C"):
            high = mid
        else:
            low = mid

    return high


def extra_time_from_cooling_h(params: SimParams, cooling_W: float):
    if cooling_W is None:
        return None

    base = run_simulation(params, mode="A")
    if base["time_to_vent_h"] is None:
        return 0.0

    p = copy.deepcopy(params)
    p.cooling_W = cooling_W
    p.cooling_start_h = 0.0
    mitigated = run_simulation(p, mode="B")

    if mitigated["time_to_vent_h"] is None:
        return max(0.0, params.duration_h - base["time_to_vent_h"])

    return max(0.0, mitigated["time_to_vent_h"] - base["time_to_vent_h"])


def extra_time_from_withdrawal_h(params: SimParams, withdrawal_kgph: float):
    if withdrawal_kgph is None:
        return None

    base = run_simulation(params, mode="A")
    if base["time_to_vent_h"] is None:
        return 0.0

    p = copy.deepcopy(params)
    p.withdrawal_kg_h = withdrawal_kgph
    p.withdrawal_start_h = 0.0
    mitigated = run_simulation(p, mode="C")

    if mitigated["time_to_vent_h"] is None:
        return max(0.0, params.duration_h - base["time_to_vent_h"])

    return max(0.0, mitigated["time_to_vent_h"] - base["time_to_vent_h"])


def decision_support(params: SimParams):
    baseline = run_simulation(params, mode="A")
    ttv = baseline["time_to_vent_h"]
    threshold = params.P_vent_bar - params.safety_margin_bar
    max_p = baseline["max_pressure_bar"]

    req_cooling = None
    req_withdrawal = None

    if ttv is None and max_p < threshold:
        status = "GREEN"
        action = "No action required"
        message = (
            "Baseline simulation stays below the configured safety-margin threshold "
            "for the requested storage period."
        )
    elif ttv is None:
        status = "YELLOW"
        action = "Prepare mitigation"
        message = (
            "The tank does not vent within the requested period, but predicted pressure "
            "enters the safety-margin region."
        )
        req_cooling = find_min_cooling_duty_W(params)
        req_withdrawal = find_min_withdrawal_kgph(params)
    else:
        status = "RED"
        action = "Cooling or withdrawal"
        message = (
            f"Baseline venting is predicted after about {ttv:.2f} h. "
            "Mitigation should be applied before the vent limit is reached."
        )
        req_cooling = find_min_cooling_duty_W(params)
        req_withdrawal = find_min_withdrawal_kgph(params)

    # For yellow status, calculate useful recommendation values too
    if status == "YELLOW" and req_cooling is None:
        req_cooling = find_min_cooling_duty_W(params)
    if status == "YELLOW" and req_withdrawal is None:
        req_withdrawal = find_min_withdrawal_kgph(params)

    return {
        "status": status,
        "message": message,
        "recommended_action": action,
        "required_cooling_W": req_cooling,
        "required_withdrawal_kgph": req_withdrawal,
    }
