import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="Smart Energy Management System (SEMS)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Styling
st.markdown("""
    <style>
    .stMetric {background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; padding: 10px;}
    .st-emotion-cache-1wivap2 {font-size: 1.2rem;}
    </style>
""", unsafe_allow_html=True)


# --- 2. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('real_data_model.pkl')
        df = pd.read_csv('processed_load_profile.csv', parse_dates=[0], index_col=0)
        return model, df
    except FileNotFoundError:
        return None, None


model, df_history = load_assets()

if model is None:
    st.error("🚨 CRITICAL: AI Model not found. Run training notebook first.")
    st.stop()

# --- 3. DIGITAL TWIN STATE ---
if 'sim_time' not in st.session_state:
    st.session_state.sim_time = df_history.index.max() - timedelta(days=2)
    st.session_state.is_running = False
    st.session_state.total_cost = 0.0
    st.session_state.solar_generated = 0.0

# Initialize Advanced Fleet
if 'fleet' not in st.session_state:
    st.session_state.fleet = [
        # Type: Critical (Ambulance/Admin), VIP (Paid Premium), Regular (Employee)
        {'id': 'EV-A1', 'model': 'Tesla Model 3', 'cap': 75, 'soc': 25, 'target': 90, 'max_kw': 11, 'type': 'VIP',
         'active': True},
        {'id': 'EV-B2', 'model': 'Nissan Leaf', 'cap': 40, 'soc': 60, 'target': 100, 'max_kw': 7, 'type': 'Regular',
         'active': True},
        {'id': 'EV-C3', 'model': 'Ford F-150', 'cap': 130, 'soc': 10, 'target': 80, 'max_kw': 19, 'type': 'Critical',
         'active': True},
        {'id': 'EV-D4', 'model': 'Hyundai Ioniq', 'cap': 77, 'soc': 85, 'target': 90, 'max_kw': 22, 'type': 'Regular',
         'active': True},
        {'id': 'EV-E5', 'model': 'BMW iX', 'cap': 105, 'soc': 40, 'target': 85, 'max_kw': 11, 'type': 'VIP',
         'active': True},
        {'id': 'EV-F6', 'model': 'Volkswagen ID.4', 'cap': 82, 'soc': 15, 'target': 80, 'max_kw': 11, 'type': 'Regular',
         'active': True},
        {'id': 'EV-G7', 'model': 'Rivian R1T', 'cap': 135, 'soc': 5, 'target': 90, 'max_kw': 11, 'type': 'Critical',
         'active': True},
        {'id': 'EV-H8', 'model': 'Chevy Bolt EUV', 'cap': 65, 'soc': 55, 'target': 100, 'max_kw': 7, 'type': 'Regular',
         'active': True},
        {'id': 'EV-I9', 'model': 'Polestar 2', 'cap': 78, 'soc': 30, 'target': 85, 'max_kw': 11, 'type': 'VIP',
         'active': True},
        {'id': 'EV-J10', 'model': 'Porsche Taycan', 'cap': 93, 'soc': 20, 'target': 80, 'max_kw': 22, 'type': 'VIP',
         'active': True},
        {'id': 'EV-K11', 'model': 'Kia EV6', 'cap': 77, 'soc': 45, 'target': 90, 'max_kw': 11, 'type': 'Regular',
         'active': True},
        {'id': 'EV-L12', 'model': 'Lucid Air', 'cap': 112, 'soc': 10, 'target': 85, 'max_kw': 19, 'type': 'VIP',
         'active': True},
        {'id': 'EV-M13', 'model': 'Mercedes EQS', 'cap': 107, 'soc': 60, 'target': 80, 'max_kw': 11, 'type': 'Regular',
         'active': True},
        {'id': 'EV-N14', 'model': 'Audi Q4 e-tron', 'cap': 82, 'soc': 35, 'target': 90, 'max_kw': 11, 'type': 'Regular',
         'active': True},
        {'id': 'EV-O15', 'model': 'Volvo XC40', 'cap': 78, 'soc': 50, 'target': 80, 'max_kw': 11, 'type': 'Regular',
         'active': True},
        {'id': 'EV-P16', 'model': 'Tesla Model S', 'cap': 100, 'soc': 12, 'target': 95, 'max_kw': 17,
         'type': 'Critical',
         'active': True},
        {'id': 'EV-Q17', 'model': 'Ford Mustang Mach-E', 'cap': 88, 'soc': 40, 'target': 85, 'max_kw': 11,
         'type': 'Regular',
         'active': True},
        {'id': 'EV-R18', 'model': 'Hyundai Kona Electric', 'cap': 64, 'soc': 25, 'target': 100, 'max_kw': 7,
         'type': 'Regular',
         'active': True},
        {'id': 'EV-S19', 'model': 'Jaguar I-PACE', 'cap': 90, 'soc': 18, 'target': 80, 'max_kw': 11, 'type': 'VIP',
         'active': True},
        {'id': 'EV-T20', 'model': 'Fisker Ocean', 'cap': 113, 'soc': 5, 'target': 90, 'max_kw': 22, 'type': 'Regular',
         'active': True},
        {'id': 'EV-U21', 'model': 'Mini Cooper SE', 'cap': 32, 'soc': 65, 'target': 100, 'max_kw': 7, 'type': 'Regular',
         'active': True},
        {'id': 'EV-V22', 'model': 'Nissan Ariya', 'cap': 87, 'soc': 22, 'target': 85, 'max_kw': 22, 'type': 'Regular',
         'active': True},
        {'id': 'EV-W23', 'model': 'Cadillac Lyriq', 'cap': 102, 'soc': 33, 'target': 80, 'max_kw': 19, 'type': 'VIP',
         'active': True},
        {'id': 'EV-X24', 'model': 'Genesis GV60', 'cap': 77, 'soc': 10, 'target': 90, 'max_kw': 11, 'type': 'Regular',
         'active': True},
        {'id': 'EV-Y25', 'model': 'Toyota bZ4X', 'cap': 71, 'soc': 48, 'target': 80, 'max_kw': 6, 'type': 'Regular',
         'active': True},
    ]


# --- 4. ADVANCED SIMULATORS (The "World" Logic) ---

def get_solar_production(timestamp, capacity_kw=50):
    """Simulates solar generation based on hour of day (Gaussian bell curve)"""
    hour = timestamp.hour + timestamp.minute / 60
    # Solar typically active between 6am (6) and 6pm (18), peak at 12
    if 6 <= hour <= 18:
        # Simple Gaussian curve approximation
        return capacity_kw * np.exp(-0.5 * ((hour - 12) / 2.5) ** 2)
    return 0.0


def get_electricity_price(timestamp):
    """Simulates Time-of-Use (ToU) Tariffs ($/kWh)"""
    hour = timestamp.hour
    # Expensive: 4PM - 9PM ($0.35), Cheap: Overnight ($0.10), Normal: ($0.18)
    if 16 <= hour <= 21:
        return 0.35
    elif 0 <= hour <= 6:
        return 0.10
    else:
        return 0.18


def get_battery_physics_rate(current_soc, max_rate):
    """Simulates CC/CV curve: Charging slows down > 80% SoC"""
    if current_soc < 80:
        return max_rate
    else:
        # Linear drop-off from 80% to 100%
        factor = (100 - current_soc) / 20.0
        return max_rate * max(0.1, factor)  # Minimum 10% trickle charge


# --- 5. SMART DLM ALGORITHM (FIXED) ---
def smart_dlm_optimizer(grid_limit, solar_kw, price, vehicles, price_threshold=0.20):
    """
    Optimizes for: 1. Grid Safety, 2. Vehicle Priority, 3. Cost Saving
    """
    # 1. Calculate Total Available Power
    total_capacity = grid_limit + solar_kw

    # FIX: Initialize with a Tuple (Power, Status) so unpacking never fails
    allocations = {v['id']: (0.0, "Waiting/Idle") for v in vehicles}

    active_evs = [v for v in vehicles if v['active'] and v['soc'] < v['target']]

    if not active_evs: return allocations, 0.0

    remaining_power = total_capacity
    total_allocated = 0.0

    # Sort EVs by Priority (Critical > VIP > Regular)
    high_price_mode = price > price_threshold
    priority_map = {'Critical': 3, 'VIP': 2, 'Regular': 1}

    # Critical first. If price is high, Regulars get de-prioritized.
    active_evs.sort(key=lambda x: priority_map[x['type']], reverse=True)

    for v in active_evs:
        # A. Physical Limit (Battery Curve)
        phys_max = get_battery_physics_rate(v['soc'], v['max_kw'])

        # B. Economic Limit (Smart Charging)
        # If Regular car & High Price & Low Solar -> Throttle to minimum
        if v['type'] == 'Regular' and high_price_mode and solar_kw < 10:
            econ_max = 2.0
            status_note = "💰 Econ Throttled"
        else:
            econ_max = phys_max
            status_note = "⚡ Fast Charging"

        # C. Allocation
        wanted = min(phys_max, econ_max)
        given = min(wanted, remaining_power)

        # Assign the Tuple (Power, Note)
        allocations[v['id']] = (given, status_note)

        remaining_power -= given
        total_allocated += given

    return allocations, total_allocated


# --- 6. AI FORECAST ENGINE ---
def get_ai_prediction(ts):
    # Construct Future Frame
    future = pd.date_range(start=ts, periods=96, freq='15min')
    feats = pd.DataFrame({
        'hour': future.hour, 'day_of_week': future.dayofweek,
        'lag_24h': [df_history['total_load_kw'].mean()] * 96,  # Mock history lookup for speed
        'lag_1h': [df_history['total_load_kw'].mean()] * 96,
        'rolling_mean_24h': [df_history['total_load_kw'].mean()] * 96
    })
    return pd.Series(model.predict(feats), index=future)


# --- 7. SIDEBAR CONTROLS ---
st.sidebar.title("🎛️ Control Room")

with st.sidebar.expander("1. Infrastructure Setup", expanded=True):
    GRID_CAPACITY = st.slider("Grid Connection Limit (kW)", 50, 300, 150)
    SOLAR_SIZE = st.slider("Solar Array Size (kW)", 0, 100, 40)

with st.sidebar.expander("2. Smart Policies", expanded=True):
    PRICE_SENSITIVITY = st.slider("Max Price Threshold ($/kWh)", 0.05, 0.50, 0.20, 0.01)
    st.caption("Regular charging throttled above this price.")

with st.sidebar.expander("3. Simulation", expanded=True):
    sim_mode = st.radio("Simulation Mode", ["Manual Step", "Auto-Play (Fast)"])
    if st.button("⏩ Jump 1 Hour"):
        st.session_state.sim_time += timedelta(hours=1)

# --- 8. MAIN DASHBOARD ---

# A. Header & Environment
curr_solar = get_solar_production(st.session_state.sim_time, SOLAR_SIZE)
curr_price = get_electricity_price(st.session_state.sim_time)
ai_forecast = get_ai_prediction(st.session_state.sim_time)

# B. Run Optimizer
allocations, total_load = smart_dlm_optimizer(
    GRID_CAPACITY, curr_solar, curr_price, st.session_state.fleet, PRICE_SENSITIVITY
)

# C. Calculate Net Grid Usage (Load - Solar)
net_grid_load = max(0, total_load - curr_solar)
grid_strain_pct = (net_grid_load / GRID_CAPACITY) * 100

# UI: Environment Row
c1, c2, c3, c4 = st.columns(4)
c1.metric("🕒 System Time", st.session_state.sim_time.strftime('%H:%M'), st.session_state.sim_time.strftime('%Y-%m-%d'))
c2.metric("☀️ Solar Generation", f"{curr_solar:.1f} kW",
          f"Offsetting {min(100, (curr_solar / total_load) * 100) if total_load > 0 else 0:.0f}% Load")
c3.metric("💲 Electricity Price", f"${curr_price:.2f} /kWh", "Peak Pricing" if curr_price > 0.20 else "Off-Peak",
          delta_color="inverse")
c4.metric("gd Grid Usage", f"{net_grid_load:.1f} kW", f"{grid_strain_pct:.1f}% Capacity",
          delta_color="inverse" if grid_strain_pct > 90 else "normal")

# D. Fleet Status Table (With Logic Feedback)
st.subheader("🚗 Live Fleet Operations")
fleet_display = []
for v in st.session_state.fleet:
    pwr, note = allocations[v['id']]
    # Bat Physics: Calculate time
    if pwr > 0:
        kwh_needed = v['cap'] * (v['target'] - v['soc']) / 100
        hrs = kwh_needed / pwr
        eta = f"{int(hrs)}h {int((hrs * 60) % 60)}m"
    else:
        eta = "Paused" if v['soc'] < v['target'] else "Done"

    fleet_display.append({
        "ID": v['id'], "Model": v['model'], "Type": v['type'],
        "SoC": v['soc'], "Target": v['target'],
        "Power (kW)": f"{pwr:.2f}", "Source": "Solar+Grid",
        "Logic State": note, "ETA": eta
    })

st.dataframe(
    pd.DataFrame(fleet_display),
    column_config={
        "SoC": st.column_config.ProgressColumn("Battery %", format="%d%%", min_value=0, max_value=100),
        "Power (kW)": st.column_config.NumberColumn("Charging Rate", format="%.2f kW")
    },
    use_container_width=True, hide_index=True
)

# E. Real-Time Analytics Charts
c_left, c_right = st.columns([2, 1])

with c_left:
    st.subheader("📉 Energy Mix & Prediction")
    # Combine actual load vs capacity
    fig = go.Figure()

    # Grid Limit Area
    fig.add_trace(
        go.Scatter(x=ai_forecast.index, y=[GRID_CAPACITY] * 96, name="Grid Limit", line=dict(color='red', dash='dash')))

    # Solar Curve (Visualizing free energy)
    solar_curve = [get_solar_production(t, SOLAR_SIZE) for t in ai_forecast.index]
    fig.add_trace(go.Scatter(x=ai_forecast.index, y=solar_curve, name="Solar Forecast", fill='tozeroy',
                             line=dict(color='#ffd700')))

    # AI Load Prediction
    fig.add_trace(
        go.Scatter(x=ai_forecast.index, y=ai_forecast.values + np.array(solar_curve), name="Predicted Total Demand",
                   line=dict(color='blue')))

    fig.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with c_right:
    st.subheader("💰 Cost & Efficiency")
    # Cost accumulation logic
    cost_now = (net_grid_load * curr_price) / 4  # Cost for 15 mins
    st.session_state.total_cost += cost_now
    st.session_state.solar_generated += (curr_solar / 4)

    st.metric("Total Session Cost", f"${st.session_state.total_cost:.2f}")
    st.metric("Solar Energy Used", f"{st.session_state.solar_generated:.1f} kWh", "Free Energy")

    # Gauge for Price
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=curr_price,
        title={'text': "Current Tariff ($)"},
        gauge={'axis': {'range': [0, 0.5]}, 'bar': {'color': "black"},
               'steps': [{'range': [0, 0.2], 'color': "lightgreen"}, {'range': [0.2, 0.5], 'color': "salmon"}]}
    ))
    fig_g.update_layout(height=300, margin=dict(t=60, b=30, l=20, r=20))
    st.plotly_chart(fig_g, use_container_width=True)


# --- 9. ROBUST AUTOMATION ENGINE ---

# Helper function to handle different Streamlit versions
def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


# Execute Auto-Play Logic
if sim_mode == "Auto-Play (Fast)":
    # 1. Update System Time
    st.session_state.sim_time += timedelta(minutes=15)

    # 2. Update Vehicle Batteries
    for v in st.session_state.fleet:
        # Only update active cars that need charging
        if v['active'] and v['soc'] < v['target']:
            # Get allocation safe-guarding against data errors
            allocation_data = allocations.get(v['id'], 0.0)

            # Check if we got a tuple (Power, Status) or just a float (0.0)
            if isinstance(allocation_data, tuple):
                pwr, _ = allocation_data
            else:
                pwr = allocation_data  # It's just a number (likely 0.0)

            # Calculate energy added in 15 mins (0.25 hours)
            if pwr > 0:
                added_kwh = pwr * 0.25
                added_soc_pct = (added_kwh / v['cap']) * 100

                # Apply update
                v['soc'] = min(v['target'], v['soc'] + added_soc_pct)

    # 3. Wait and Rerun
    time.sleep(0.5)  # Animation speed (0.5 seconds per frame)
    safe_rerun()

    st.rerun()