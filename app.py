import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, time as dt_time
import time
import random
import io
import joblib  # <--- ADD THIS
import xgboost as xgb # <--- ADD THIS

from fleet_manager import generate_fleet

# --- 1. SYSTEM CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="SEMS: Enterprise Digital Twin",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load CSS
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ style.css not found. UI may look unstyled.")


load_css("style.css")


# --- NEW: LOAD AI ASSETS ---
@st.cache_resource
def load_prediction_assets():
    """
    Loads the trained XGBoost model and historical data for predictions.
    Generated from '01_EV_Load_Forecasting_Model.ipynb'.
    """
    try:
        # Load the model
        model = joblib.load('real_data_model.pkl')

        # Load the history data
        # We parse dates to ensure the index is datetime objects
        df_history = pd.read_csv('processed_load_profile.csv', index_col=0, parse_dates=True)

        return model, df_history
    except FileNotFoundError:
        # Graceful fallback if files are missing
        return None, None


# Load them into global variables
ai_model, ai_history = load_prediction_assets()


# --- 2. ADVANCED PHYSICS MODELS ---

class BatteryPhysics:
    """
    Simulates Lithium-Ion physics: Degradation, Thermal Throttling,
    and CC-CV (Constant Current - Constant Voltage) charging curves.
    """

    @staticmethod
    def get_max_charge_rate(soc, max_kw, health_pct, temp_c=25):
        """
        Calculates realistic charge rate based on:
        1. SOC (Slows down > 80%)
        2. SOH (Degraded batteries charge slower)
        3. Temperature (Cold/Hot batteries throttle)
        """
        # A. Base Physics (CC-CV Curve)
        if soc < 80:
            rate = max_kw
        else:
            # Linear drop from 80% to 100%
            rate = max_kw * ((100 - soc) / 20.0)

        # B. Health Impact (Old batteries have higher internal resistance)
        rate *= (health_pct / 100.0)

        # C. Temperature Impact (Simple bell curve around 25°C)
        # Efficiency drops if temp is far from 25°C
        temp_efficiency = np.exp(-0.5 * ((temp_c - 25) / 10.0) ** 2)
        rate *= max(0.5, temp_efficiency)  # Min 50% speed even in bad temps

        return max(0.0, rate)

    @staticmethod
    def calculate_degradation(current_soh, cycles_added, depth_of_discharge):
        """
        Estimates battery health loss.
        Cycle Life Model: Li-ion typically lasts 1000-2000 cycles.
        """
        # Simple linear degradation model for simulation
        # 0.005% loss per full cycle equivalent
        degradation = cycles_added * 0.005 * (1 + (depth_of_discharge / 100))
        return max(0.0, current_soh - degradation)


class EnvironmentModel:
    """Generates weather, solar, and pricing data."""

    @staticmethod
    def get_solar_kw(timestamp, installed_cap_kw):
        """Gaussian Solar Generation Model"""
        hour = timestamp.hour + timestamp.minute / 60.0
        # Peak at 13:00, Standard Deviation 2.5 hours
        intensity = np.exp(-0.5 * ((hour - 13) / 2.5) ** 2)
        # Add random cloud cover noise (10% variance)
        noise = random.uniform(0.9, 1.1)
        return max(0.0, installed_cap_kw * intensity * noise)

    @staticmethod
    def get_grid_price(timestamp):
        """ToU (Time of Use) Pricing Model ($/kWh)"""
        h = timestamp.hour
        # Critical Peak: 17:00 - 20:00
        if 17 <= h < 20:
            return 0.45
        # High Day: 08:00 - 17:00
        elif 8 <= h < 17:
            return 0.22
        # Overnight: 20:00 - 08:00
        else:
            return 0.12


def get_ai_forecast(current_time, model, history_df):
    """
    Generates a 24-hour load forecast using the loaded XGBoost model.
    """
    if model is None or history_df is None:
        return pd.DataFrame()

    # 1. Create Future Timeframe (Next 24 hours = 96 steps of 15 mins)
    future_index = pd.date_range(start=current_time, periods=96, freq='15min')

    # 2. Construct Feature DataFrame
    # We must match the EXACT features used in the notebook training
    df_future = pd.DataFrame(index=future_index)
    df_future['hour'] = df_future.index.hour
    df_future['day_of_week'] = df_future.index.dayofweek

    # 3. Populate Lag Features (Heuristic for Simulation)
    # In a perfect scenario, we look up exact historical timestamps.
    # For this simulation, we take the average load profile from the CSV
    # to serve as our "lag" inputs, ensuring the model always has data to read.

    # Get a baseline load from history to simulate 'past behavior'
    avg_load = history_df['total_load_kw'].mean()

    # Create synthetic lags based on the notebook's logic
    # lag_24h: What was the load this time yesterday?
    # lag_1h: What was the load 1 hour ago?

    # We add some random variance to make it look realistic
    df_future['lag_24h'] = [avg_load * random.uniform(0.8, 1.2) for _ in range(96)]
    df_future['lag_1h'] = [avg_load * random.uniform(0.9, 1.1) for _ in range(96)]
    df_future['rolling_mean_24h'] = avg_load  # Simplified rolling mean

    # 4. Select Features in Order
    features = ['hour', 'day_of_week', 'lag_24h', 'lag_1h', 'rolling_mean_24h']

    # 5. Run Prediction
    try:
        predictions = model.predict(df_future[features])
        df_future['predicted_load'] = predictions
        return df_future
    except Exception as e:
        # Fallback if feature names mismatch
        return pd.DataFrame()

# --- 3. SESSION STATE MANAGEMENT ---

# --- CORRECTED INIT FUNCTION ---
def init_session_state():
    # ... (Keep time and financial checks same as before) ...
    if 'sim_time' not in st.session_state:
        st.session_state.sim_time = datetime(2024, 6, 1, 6, 0, 0)

    if 'is_running' not in st.session_state:
        st.session_state.is_running = False

    if 'financials' not in st.session_state:
        st.session_state.financials = {
            'cost_smart': 0.0, 'cost_dumb': 0.0,
            'solar_kwh': 0.0, 'grid_kwh': 0.0
        }

    if 'history' not in st.session_state:
        st.session_state.history = []

    # --- FIX IS HERE ---
    if 'fleet' not in st.session_state:
        # Generate 300 random cars using the new file
        st.session_state.fleet = generate_fleet(num_cars=300)


init_session_state()


# --- 4. CORE ALGORITHMS ---

def run_simulation_step(grid_cap, solar_cap, price_threshold):
    """
    Executes one 15-minute time step of the simulation.
    Returns: A dictionary of current stats for plotting.
    """
    ts = st.session_state.sim_time

    # 1. Environment
    solar_gen = EnvironmentModel.get_solar_kw(ts, solar_cap)
    price = EnvironmentModel.get_grid_price(ts)

    # 2. Logic Preparation
    # We simulate TWO scenarios simultaneously:
    # Scenario A: Smart DLM (The actual simulation)
    # Scenario B: Dumb Charging (What would happen without logic)

    total_smart_load = 0.0
    total_dumb_load = 0.0

    # Priority sorting: Critical > VIP > Regular
    priority_map = {'Critical': 3, 'VIP': 2, 'Regular': 1}
    active_fleet = [v for v in st.session_state.fleet if v['connected']]
    active_fleet.sort(key=lambda x: priority_map[x['type']], reverse=True)

    # --- SCENARIO A: SMART CHARGING (DLM) ---
    remaining_grid = grid_cap + solar_gen

    for car in active_fleet:
        # Physics Check
        phys_max = BatteryPhysics.get_max_charge_rate(
            car['soc'], car['max_kw'], car['soh']
        )

        # Logic Check
        if car['soc'] >= car['target_soc']:
            alloc = 0.0
            status = "Done"
        else:
            # DLM Rule: Throttle Regular cars if price is high or solar is low
            if car['type'] == 'Regular' and price > price_threshold and solar_gen < 5:
                econ_limit = 1.4  # Minimum trickle charge
                status = "Throttled ($)"
            else:
                econ_limit = phys_max
                status = "Charging"

            # Grid Constraint
            alloc = min(phys_max, econ_limit, remaining_grid)

        # Apply Charge
        if alloc > 0:
            kwh_added = alloc * 0.25  # 15 mins
            soc_added = (kwh_added / car['cap_kwh']) * 100
            car['soc'] = min(100, car['soc'] + soc_added)

            # Physics: Degrade Battery
            # Cycle = kWh / Cap. Depth = (100 - SOC).
            deg = BatteryPhysics.calculate_degradation(
                car['soh'], (kwh_added / car['cap_kwh']), (100 - car['soc'])
            )
            car['soh'] = deg

            remaining_grid -= alloc

        car['current_kw'] = alloc
        car['status'] = status
        total_smart_load += alloc

    # --- SCENARIO B: DUMB CHARGING (SHADOW) ---
    for car in active_fleet:
        # In dumb mode, everyone pulls max power instantly regardless of price/grid
        if car['shadow_soc'] < car['target_soc']:
            phys_max = BatteryPhysics.get_max_charge_rate(
                car['shadow_soc'], car['max_kw'], car['soh']
            )
            total_dumb_load += phys_max
            # Update shadow battery
            kwh_added = phys_max * 0.25
            soc_added = (kwh_added / car['cap_kwh']) * 100
            car['shadow_soc'] = min(100, car['shadow_soc'] + soc_added)

    # 3. Financial Calculations
    net_smart = max(0, total_smart_load - solar_gen)
    net_dumb = max(0, total_dumb_load - solar_gen)

    cost_smart = net_smart * price * 0.25
    cost_dumb = net_dumb * price * 0.25

    # Update Totals
    st.session_state.financials['cost_smart'] += cost_smart
    st.session_state.financials['cost_dumb'] += cost_dumb
    st.session_state.financials['solar_kwh'] += (min(total_smart_load, solar_gen) * 0.25)
    st.session_state.financials['grid_kwh'] += (net_smart * 0.25)

    # 4. Log Data
    log_entry = {
        'time': ts,
        'solar': solar_gen,
        'price': price,
        'smart_load': total_smart_load,
        'dumb_load': total_dumb_load,
        'grid_limit': grid_cap,
        'grid_usage': net_smart
    }
    st.session_state.history.append(log_entry)

    # Keep history manageable (last 2000 points ~ 20 days simulated)
    if len(st.session_state.history) > 2000:
        st.session_state.history.pop(0)


# --- 5. UI COMPONENTS ---

def render_sidebar():
    st.sidebar.title("🎛️ Control Panel")

    # A. SIMULATION CONTROLS
    st.sidebar.subheader("Simulation State")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.session_state.is_running:
            if st.button("⏸️ PAUSE", type="primary", use_container_width=True):
                st.session_state.is_running = False
                st.rerun()
        else:
            if st.button("▶️ START", type="primary", use_container_width=True):
                st.session_state.is_running = True
                st.rerun()

    with col2:
        if st.button("🔄 RESET", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    # B. MANUAL TIME MANAGEMENT
    st.sidebar.markdown("---")
    st.sidebar.subheader("🕒 Time Management")

    if not st.session_state.is_running:
        with st.sidebar.form("time_set_form"):
            new_date = st.date_input("Date", value=st.session_state.sim_time.date())
            new_time = st.time_input("Time", value=st.session_state.sim_time.time())
            submit = st.form_submit_button("Set System Time")

            if submit:
                st.session_state.sim_time = datetime.combine(new_date, new_time)
                st.success("Time updated successfully.")
                time.sleep(0.5)
                st.rerun()
    else:
        st.sidebar.info("⏸️ Pause simulation to manually adjust time.")
        st.sidebar.metric("Current Time", st.session_state.sim_time.strftime("%H:%M"))

    # C. PARAMETERS (UPDATED FOR 50 CARS)
    st.sidebar.markdown("---")
    with st.sidebar.expander("⚙️ Infrastructure Settings", expanded=True):
        # [CHANGE] Scaled down for 50 cars (Default 250 kW)
        grid_cap = st.slider("Grid Transformer Limit (kW)", 20, 1000, 250,
                             help="Physical limit of the local grid connection.")

        # [CHANGE] Scaled down Solar (Default 80 kW)
        solar_cap = st.slider("Solar Array Capacity (kW)", 0, 500, 80,
                              help="Peak output of installed solar panels.")

        price_sens = st.slider("Price Threshold ($/kWh)", 0.05, 1.00, 0.25,
                               help="Above this price, non-critical cars are throttled.")

    return grid_cap, solar_cap, price_sens


def render_dashboard(grid_cap, solar_cap, price_thresh):
    # --- Top Metrics Row ---
    fin = st.session_state.financials
    savings = fin['cost_dumb'] - fin['cost_smart']

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Time", st.session_state.sim_time.strftime("%b %d, %H:%M"))
    m2.metric("Total Savings", f"${savings:,.2f}",
              delta=f"{(savings / fin['cost_dumb'] * 100) if fin['cost_dumb'] else 0:.1f}% vs Dumb")
    m3.metric("Solar Energy", f"{fin['solar_kwh']:.1f} kWh", "Free Energy")
    m4.metric("Grid Energy", f"{fin['grid_kwh']:.1f} kWh", f"-${fin['cost_smart']:.2f}")

    # Tabs for detailed views
    tab_main, tab_analytics, tab_fleet = st.tabs(["📊 Live Operation", "📈 Deep Analysis", "🚗 Fleet Health"])

    # --- TAB 1: LIVE OPERATION ---
    with tab_main:
        c_left, c_right = st.columns([3, 1])

        with c_left:
            st.subheader("Real-Time Power Flow")
            if st.session_state.history:
                df = pd.DataFrame(st.session_state.history)
                fig = go.Figure()

                # Overload Zone (Visual Aid)
                fig.add_hrect(y0=grid_cap, y1=grid_cap * 1.5, line_width=0, fillcolor="red", opacity=0.1,
                              annotation_text="Overload Zone")

                # Traces
                # 1. Solar Generation (Yellow Area)
                fig.add_trace(go.Scatter(x=df['time'], y=df['solar'], name="Solar Gen", fill='tozeroy',
                                         line=dict(color='#ffcc00')))

                # 2. Net Grid Usage (Blue Line) - UPDATED to show 'grid_usage' so it respects the limit visually
                fig.add_trace(go.Scatter(x=df['time'], y=df['grid_usage'], name="Grid Draw (Net)",
                                         line=dict(color='#0066cc', width=2)))

                # 3. Grid Limit (Red Dashed Line)
                fig.add_trace(go.Scatter(x=df['time'], y=[grid_cap] * len(df), name="Grid Cap",
                                         line=dict(color='red', dash='dash')))

                fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0), hovermode="x unified",
                                  yaxis_title="Power (kW)")

                # [FIX] Added unique key
                st.plotly_chart(fig, use_container_width=True, key="chart_live_power_flow")
            else:
                st.info("Waiting for simulation data... Press START in the sidebar.")

        with c_right:
            st.subheader("Current Environment")
            curr_price = EnvironmentModel.get_grid_price(st.session_state.sim_time)

            # Gauge Color Logic
            if curr_price <= 0.15:
                gauge_color = "#00ff9d"
            elif curr_price <= 0.30:
                gauge_color = "#ffcc00"
            else:
                gauge_color = "#ff4b4b"

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=curr_price,
                number={'suffix': " $/kWh", 'font': {'size': 20, 'color': "white", 'family': "Inter"}},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "GRID TARIFF", 'font': {'size': 12, 'color': "#a0a0a0"}},
                gauge={
                    'axis': {'range': [0, 0.6], 'visible': False},
                    'bar': {'color': gauge_color, 'thickness': 0.8},
                    'bgcolor': "rgba(255,255,255,0.05)",
                    'borderwidth': 0,
                    'threshold': {'line': {'color': "white", 'width': 3}, 'thickness': 0.8, 'value': price_thresh}
                }
            ))
            fig_g.update_layout(height=180, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)',
                                font={'family': "Inter"})

            # [FIX] Added unique key
            st.plotly_chart(fig_g, use_container_width=True, key="chart_live_gauge")

            # Status Pills
            solar_now = EnvironmentModel.get_solar_kw(st.session_state.sim_time, solar_cap)
            is_active = curr_price > price_thresh
            st.markdown(f"""
            <div style="display: flex; flex-direction: column; gap: 10px; margin-top: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(255,255,255,0.05); padding: 8px 12px; border-radius: 8px;">
                    <span style="color: #a0a0a0; font-size: 0.9em;">☀️ Solar Input</span>
                    <span style="color: #ffd700; font-weight: 600; font-family: 'JetBrains Mono';">{solar_now:.1f} kW</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(255,255,255,0.05); padding: 8px 12px; border-radius: 8px;">
                    <span style="color: #a0a0a0; font-size: 0.9em;">⚡ Logic State</span>
                    <span class="status-badge {'status-throttled' if is_active else 'status-charging'}">
                        {'ACTIVE (THROTTLING)' if is_active else 'PASSIVE (FULL SPEED)'}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # --- TAB 2: ANALYTICS ---
    with tab_analytics:
        st.header("📊 Deep Dive Analytics")

        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)

            # 1. AI FORECASTING
            st.subheader("🤖 AI Load Forecasting (Next 24h)")

            # Call prediction logic
            forecast_df = get_ai_forecast(st.session_state.sim_time, ai_model, ai_history)

            if not forecast_df.empty:
                fig_cast = go.Figure()

                # Trace: AI Prediction
                fig_cast.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['predicted_load'],
                                              mode='lines', name='AI Predicted Demand',
                                              line=dict(color='#ab63fa', width=3, dash='solid')))

                # Trace: Grid Limit
                fig_cast.add_trace(go.Scatter(x=forecast_df.index, y=[grid_cap] * len(forecast_df),
                                              mode='lines', name='Grid Capacity Limit',
                                              line=dict(color='#ff4b4b', dash='dash')))

                # Trace: Actuals (Comparison) - Optional, only shows if dates match CSV
                try:
                    actuals_df = ai_history.loc[ai_history.index.isin(forecast_df.index)]
                    if not actuals_df.empty:
                        fig_cast.add_trace(go.Scatter(x=actuals_df.index, y=actuals_df['total_load_kw'],
                                                      mode='lines', name='Historical Actuals',
                                                      line=dict(color='gray', width=2, dash='dot'), opacity=0.7))
                except:
                    pass

                # Danger Markers
                danger = forecast_df[forecast_df['predicted_load'] > grid_cap]
                if not danger.empty:
                    fig_cast.add_trace(go.Scatter(x=danger.index, y=danger['predicted_load'],
                                                  mode='markers', name='Predicted Overload',
                                                  marker=dict(color='red', size=8, symbol='x')))

                fig_cast.update_layout(height=350, xaxis_title="Future Time", yaxis_title="Power (kW)",
                                       hovermode="x unified")

                # [FIX] Added unique key
                st.plotly_chart(fig_cast, use_container_width=True, key="chart_ai_forecast")
            else:
                st.info("Model not active. Ensure 'real_data_model.pkl' is available.")

            st.markdown("---")

            # 2. SANKEY & COST
            col_c, col_d = st.columns([1, 1])

            with col_c:
                st.markdown("#### ⚡ Current Power Flow")
                last_step = st.session_state.history[-1]
                fleet_df = pd.DataFrame(st.session_state.fleet)
                usage = fleet_df.groupby('type')['current_kw'].sum()

                fig_sankey = go.Figure(data=[go.Sankey(
                    node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5),
                              label=["Grid", "Solar", "System", "VIP", "Critical", "Regular"],
                              color=["#ff4b4b", "#ffd700", "#2c3e50", "#9b59b6", "#e74c3c", "#3498db"]),
                    link=dict(source=[0, 1, 2, 2, 2], target=[2, 2, 3, 4, 5],
                              value=[last_step['grid_usage'], last_step['solar'], usage.get('VIP', 0),
                                     usage.get('Critical', 0), usage.get('Regular', 0)]))])

                # [FIX] Added unique key
                st.plotly_chart(fig_sankey, use_container_width=True, key="chart_sankey")

            with col_d:
                st.markdown("#### 💰 Cumulative Cost")
                df['cum_smart'] = (df['grid_usage'] * df['price'] * 0.25).cumsum()
                df['cum_dumb'] = (np.maximum(0, df['dumb_load'] - df['solar']) * df['price'] * 0.25).cumsum()

                fig_cost = go.Figure()
                fig_cost.add_trace(go.Scatter(x=df['time'], y=df['cum_dumb'], name="Unmanaged",
                                              line=dict(color='#ff4b4b', dash='dot')))
                fig_cost.add_trace(
                    go.Scatter(x=df['time'], y=df['cum_smart'], name="Smart DLM", line=dict(color='#00ff9d', width=3)))

                # [FIX] Added unique key
                st.plotly_chart(fig_cost, use_container_width=True, key="chart_cost_compare")

        else:
            st.info("👋 Simulation is waiting to start. Click '▶️ START' in the sidebar.")

    # --- TAB 3: FLEET HEALTH ---
    with tab_fleet:
        st.subheader("Vehicle Telemetry")
        fleet_df = pd.DataFrame(st.session_state.fleet)

        st.markdown("#### 🔋 State of Charge Map")
        fig_map = px.bar(fleet_df, x='id', y='soc', color='type',
                         color_discrete_map={'Critical': '#e74c3c', 'VIP': '#f1c40f', 'Regular': '#3498db'})
        fig_map.add_hline(y=90, line_dash="dot", annotation_text="Target")

        # [FIX] Added unique key
        st.plotly_chart(fig_map, use_container_width=True, key="chart_fleet_health_map")

        st.markdown("#### 📋 Live Status Board")
        display_df = fleet_df[['id', 'model', 'type', 'soc', 'current_kw', 'status']].copy()
        display_df['soc'] = display_df['soc'].apply(lambda x: f"{x:.1f}%")
        display_df['current_kw'] = display_df['current_kw'].apply(lambda x: f"{x:.1f} kW")
        st.dataframe(display_df, use_container_width=True)


# --- 6. MAIN EXECUTION LOOP ---

def main():
    # Render Sidebar
    grid_cap, solar_cap, price_sens = render_sidebar()

    # Render Main UI
    render_dashboard(grid_cap, solar_cap, price_sens)

    # Logic Loop
    if st.session_state.is_running:
        # A. Update Time
        st.session_state.sim_time += timedelta(minutes=15)

        # B. Run Physics & Logic
        run_simulation_step(grid_cap, solar_cap, price_sens)

        # C. Rerun Trigger (Visual Refresh)
        time.sleep(0.15)  # Controls animation speed
        st.rerun()


if __name__ == "__main__":
    main()