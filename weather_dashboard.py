import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Kathmandu AI Weather",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === SESSION STATE ===
for key in ['forecast_generated', 'current_forecast_24h', 'detailed_forecast', 
           'last_forecast_run', 'user_location']:
    if key not in st.session_state:
        st.session_state[key] = None if key not in ['forecast_generated', 'user_location'] else False

# === CACHING ===
@st.cache_resource
def load_predictor():
    return AdvancedWeatherPredictor()

@st.cache_resource
def load_label_encoder():
    try:
        return joblib.load("enhanced_weather_label_encoder.pkl")
    except:
        return None

# === ENHANCED CSS ===
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f3f7fb 0%, #e7eff6 100%);
        font-family: 'Segoe UI', sans-serif;
    }
    .header-title {
        font-size: 3.2rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #4F87FF, #6AC9FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.9rem;
        font-weight: 700;
        color: #4F87FF;
        border-bottom: 3px solid #6AC9FF;
        padding-bottom: 8px;
        margin: 2rem 0 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #D1E4FF 0%, #B8D4FF 100%);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-left: 5px solid #6AC9FF;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {transform: translateY(-5px);}
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4F87FF, #6AC9FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .forecast-card {
        background: #E1F1FF;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border-left: 5px solid #6AC9FF;
        margin: 0.5rem 0;
        color: #2A3D66;
        transition: all 0.3s ease;
    }
    .forecast-card:hover {background: #D4E7FF; transform: scale(1.02);}
    .condition-badge {
        background: #6AC9FF;
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fbff 0%, #e8f2ff 100%);
    }
    .metric-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        color: #4F87FF;
        font-weight: bold;
    }
    .param-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# === WEATHER CONDITION DISPLAY ===
def get_weather_display(condition: str) -> str:
    """Return clean text display for weather conditions"""
    return condition

# === UTILITY FUNCTIONS ===
def calculate_apparent_temperature(temperature, humidity, wind_speed):
    heat_index = (0.5 * (temperature + 61.0 + ((temperature - 68.0) * 1.2) + (humidity * 0.094)))
    apparent_temp = np.where(
        temperature >= 20,
        np.where(temperature >= 80, heat_index + ((temperature - 80) * 0.1), heat_index),
        13.12 + 0.6215 * temperature - 11.37 * (wind_speed ** 0.16) + 0.3965 * temperature * (wind_speed ** 0.16)
    )
    return apparent_temp

def calculate_apparent_temperature_single(temperature, humidity, wind_speed):
    if temperature >= 20:
        heat_index = (0.5 * (temperature + 61.0 + ((temperature - 68.0) * 1.2) + (humidity * 0.094)))
        return heat_index + ((temperature - 80) * 0.1) if temperature >= 80 else heat_index
    else:
        return 13.12 + 0.6215 * temperature - 11.37 * (wind_speed ** 0.16) + 0.3965 * temperature * (wind_speed ** 0.16)

def calculate_dew_point(temperature, humidity):
    return temperature - ((100 - humidity) / 5)

# === ADVANCED PREDICTOR CLASS ===
class AdvancedWeatherPredictor:
    def __init__(self):
        self.base_cols = ['temperature_2m', 'precipitation', 'wind_speed_10m', 
                         'relative_humidity_2m', 'cloud_cover', 'pressure_msl']
        self.models = {}
        self.features = []
        self.le = load_label_encoder()
        self._load_models()

    def _load_models(self):
        targets = self.base_cols + ['weather_condition_encoded']
        for t in targets:
            try:
                self.models[t] = joblib.load(f"enhanced_weather_model_{t}.pkl")
            except:
                pass
        try:
            self.features = joblib.load("enhanced_model_features.pkl")
        except:
            self.features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos']

    def _ensure_columns(self, df):
        required = ['time'] + self.base_cols
        for col in required:
            if col not in df.columns:
                df[col] = 0.0 if col != 'time' else pd.to_datetime('now')
        return df

    def _get_seasonal_adjustment(self, month):
        adjustments = {1:-3, 2:-1, 3:2, 4:4, 5:5, 6:3, 7:1, 8:1, 9:2, 10:1, 11:-1, 12:-2}
        return adjustments.get(month, 0)
    
    def calculate_heat_index(self, temperature, humidity):
        hi = (0.5 * (temperature + 61.0 + ((temperature - 68.0) * 1.2) + (humidity * 0.094)))
        return np.where(temperature >= 80, hi + ((temperature - 80) * 0.1), hi)

    def _add_temporal_features(self, df):
        df['hour'] = df['time'].dt.hour
        df['month'] = df['time'].dt.month
        df['day_of_year'] = df['time'].dt.dayofyear
        df['weekday'] = df['time'].dt.weekday
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year']/365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year']/365)
        df['diurnal_phase'] = 6 * np.sin(2 * np.pi * (df['hour'] - 14) / 24)
        df['seasonal_adj'] = df['month'].apply(self._get_seasonal_adjustment)
        df['monsoon_effect'] = df['month'].apply(lambda x: 0.4 if x in [6,7,8,9] else 0.1)
        if all(col in df.columns for col in ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m']):
            df['apparent_temperature'] = calculate_apparent_temperature(
                df['temperature_2m'], df['relative_humidity_2m'], df['wind_speed_10m']
            )
            df['heat_index'] = self.calculate_heat_index(df['temperature_2m'], df['relative_humidity_2m'])
        return df

    def _add_dynamic_features(self, df, lags=[1, 3, 6, 12, 24, 48], windows=[3, 6, 12, 24, 48]):
        for col in self.base_cols:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    if lag > 1:
                        df[f'{col}_diff_{lag}'] = df[col].diff(lag)
        for col in self.base_cols:
            if col in df.columns:
                for window in windows:
                    shifted = df[col].shift(1)
                    df[f'{col}_rolling_mean_{window}'] = shifted.rolling(window=window, min_periods=1).mean()
                    df[f'{col}_rolling_std_{window}'] = shifted.rolling(window=window, min_periods=1).std()
                    df[f'{col}_rolling_min_{window}'] = shifted.rolling(window=window, min_periods=1).min()
                    df[f'{col}_rolling_max_{window}'] = shifted.rolling(window=window, min_periods=1).max()
                    df[f'{col}_rolling_range_{window}'] = (
                        df[f'{col}_rolling_max_{window}'] - df[f'{col}_rolling_min_{window}']
                    )
                    df[f'{col}_ema_{window}'] = shifted.ewm(span=window, min_periods=1).mean()
        return df

    def predict_condition(self, row):
        if self.le and 'weather_condition_encoded' in self.models:
            model = self.models['weather_condition_encoded']
            X_row = {f: row[f] if f in row else 0.0 for f in self.features}
            X = pd.DataFrame([X_row])
            try:
                pred_encoded = model.predict(X)[0]
                return self.le.inverse_transform([int(pred_encoded)])[0]
            except:
                pass
        
        t = row.get('temperature_2m', 20)
        p = row.get('precipitation', 0)
        c = row.get('cloud_cover', 50)
        w = row.get('wind_speed_10m', 5)
        h = row.get('relative_humidity_2m', 70)
        
        if p > 8.0 and c > 85 and w > 25: return "Thunderstorm"
        if p > 5.0: return "Heavy Rain"
        if p > 2.0: return "Moderate Rain"
        if p > 0.5: return "Light Rain"
        if p > 0.1: return "Drizzle"
        if t < 0 and p > 0.1: return "Snow"
        if h > 90 and c > 80 and w < 5: return "Fog"
        if c > 80: return "Overcast"
        if c > 60: return "Mostly Cloudy"
        if c > 30: return "Partly Cloudy"
        if w > 30: return "Windy"
        if w > 20: return "Breezy"
        if t > 28: return "Hot"
        if t > 22: return "Warm"
        if t > 15 and c < 30: return "Clear Sky"
        if t > 8: return "Cool"
        return "Cold"

    def generate_forecast(self, hist_df, hours=72, start_time=None):
        if start_time is None:
            start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        state = self._ensure_columns(hist_df.copy()).tail(200).reset_index(drop=True)
        preds = []
        for h in range(hours):
            current_time = start_time + timedelta(hours=h + 1)
            new_row_template = state.iloc[-1:].copy()
            new_row_template['time'] = current_time
            for target in self.base_cols:
                if target in new_row_template.columns:
                    new_row_template[target] = np.nan 
            state = pd.concat([state, new_row_template], ignore_index=True)
            state = self._add_temporal_features(state)
            state = self._add_dynamic_features(state) 

            X_row = {}
            for f in self.features:
                X_row[f] = state[f].iloc[-1] if f in state.columns else 0.0
            X = pd.DataFrame([X_row])

            pred = {'time': current_time}
            for target in self.base_cols:
                model = self.models.get(target)
                val = np.nan
                if model is not None:
                    try:
                        y_pred = model.predict(X)[0]
                        val = np.expm1(y_pred) if 'precipitation' in target else float(y_pred)
                    except:
                        pass
                if np.isnan(val) or model is None:
                    hour = current_time.hour
                    month = current_time.month
                    base_temp = 18 + 8 * np.sin(2 * np.pi * (month - 3) / 12)
                    diurnal = 7 * np.sin(2 * np.pi * (hour - 14) / 24)
                    if target == 'temperature_2m':
                        val = base_temp + diurnal + np.random.normal(0, 1.5)
                    elif target == 'precipitation':
                        val = max(0, np.random.exponential(0.8 if month in [6,7,8,9] else 0.15))
                    elif target == 'relative_humidity_2m':
                        val = np.clip(90 - (pred.get('temperature_2m', 20) - 15)*2 + np.random.normal(0, 10), 30, 98)
                    elif target == 'wind_speed_10m':
                        val = 3 + 5*np.random.random()
                    elif target == 'cloud_cover':
                        val = np.clip(40 + 40*np.random.random() + 30*(pred.get('precipitation', 0)>1), 0, 100)
                    elif target == 'pressure_msl':
                        val = 868 + 3*np.sin(2*np.pi*hour/24) + np.random.normal(0, 2)
                if 'precipitation' in target: val = max(0, val)
                if target in ['cloud_cover', 'relative_humidity_2m']: val = np.clip(val, 0, 100)
                pred[target] = val

            pred['weather_condition'] = self.predict_condition(pred)
            pred['apparent_temperature'] = calculate_apparent_temperature_single(
                pred['temperature_2m'], pred['relative_humidity_2m'], pred['wind_speed_10m']
            )
            preds.append(pred)
            for k, v in pred.items():
                if k in self.base_cols:
                    state.iloc[-1, state.columns.get_loc(k)] = v
            state = state.tail(200).reset_index(drop=True)
        return pd.DataFrame(preds)

# === DATA LOADING ===
@st.cache_data(ttl=3600)
def load_historical():
    try:
        df = pd.read_csv("open-meteo-27.75N85.50E1293m.csv")
        df['time'] = pd.to_datetime(df['time'])
        if all(col in df.columns for col in ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m']):
            df['apparent_temperature'] = calculate_apparent_temperature(
                df['temperature_2m'], df['relative_humidity_2m'], df['wind_speed_10m']
            )
        return df.sort_values('time').reset_index(drop=True)
    except:
        dates = pd.date_range("2023-01-01", periods=20000, freq="H")
        df = pd.DataFrame({'time': dates})
        base_temp = 18 + 8 * np.sin(2 * np.pi * (dates.month - 3) / 12)
        diurnal = 7 * np.sin(2 * np.pi * (dates.hour - 14) / 24)
        df['temperature_2m'] = base_temp + diurnal + np.random.normal(0, 1.2, len(dates))
        monsoon_months = [6, 7, 8, 9]
        precip_base = np.where(dates.month.isin(monsoon_months), 0.8, 0.15)
        df['precipitation'] = np.random.exponential(precip_base, len(dates))
        df['relative_humidity_2m'] = np.clip(70 + 25 * np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 8, len(dates)), 30, 98)
        df['wind_speed_10m'] = 3 + 4 * np.random.random(len(dates))
        df['cloud_cover'] = np.clip(45 + 35 * np.random.random(len(dates)) + 25 * (df['precipitation'] > 1), 0, 100)
        df['pressure_msl'] = 868 + 2 * np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 1.5, len(dates))
        df['apparent_temperature'] = calculate_apparent_temperature(
            df['temperature_2m'], df['relative_humidity_2m'], df['wind_speed_10m']
        )
        return df

@st.cache_data(ttl=300)
def get_current():
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": 27.7017, "longitude": 85.3206,
            "current": "temperature_2m,relative_humidity_2m,precipitation,cloud_cover,wind_speed_10m,surface_pressure",
            "timezone": "Asia/Kathmandu"
        }, timeout=10).json()
        c = r['current']
        return {
            'temperature_2m': c['temperature_2m'],
            'relative_humidity_2m': c['relative_humidity_2m'],
            'precipitation': c['precipitation'],
            'cloud_cover': c['cloud_cover'],
            'wind_speed_10m': c['wind_speed_10m'],
            'pressure_msl': c['surface_pressure'],
            'timestamp': datetime.now()
        }
    except:
        return {
            'temperature_2m': 19.5, 'relative_humidity_2m': 72, 'precipitation': 0.0,
            'cloud_cover': 50, 'wind_speed_10m': 6.5, 'pressure_msl': 868,
            'timestamp': datetime.now()
        }

# === MAIN APP ===
def main():
    st.markdown("<div class='header-title'>Kathmandu AI Weather Forecast</div>", unsafe_allow_html=True)
    predictor = load_predictor()
    hist = load_historical()
    current = get_current()

    # === SIDEBAR ===
    with st.sidebar:
        st.markdown("### Weather AI")
        st.markdown("---")
        page = st.radio("Menu", ["Dashboard", "Detailed Forecast", "Historical Data", "About"], label_visibility="collapsed")
        st.markdown("---")
        st.subheader("Location")
        st.info("Kathmandu, Nepal")
        st.markdown("**Coordinates:** 27.7172°N, 85.3240°E")
        st.markdown("**Elevation:** 1,400 meters")
        st.markdown("---")
        st.caption(f"Last update: {current['timestamp'].strftime('%Y-%m-%d %H:%M')}")
        if st.button("Refresh All Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # === DASHBOARD ===
    if page == "Dashboard":
        now = datetime.now()
        next_refresh_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        # Initialize session state if needed
        if 'last_forecast_run' not in st.session_state:
            st.session_state.last_forecast_run = now - timedelta(hours=2)  # Force refresh
        
        if (st.session_state.current_forecast_24h is None or 
            now > st.session_state.last_forecast_run):
            with st.spinner("Generating 24-hour forecast..."):
                fc_24h = predictor.generate_forecast(hist, hours=24)
                st.session_state.current_forecast_24h = fc_24h
                st.session_state.last_forecast_run = next_refresh_time
        else:
            fc_24h = st.session_state.current_forecast_24h

        st.markdown("<div class='section-header'>Current Weather in Kathmandu</div>", unsafe_allow_html=True)
        apparent_temp = calculate_apparent_temperature_single(
            current['temperature_2m'], current['relative_humidity_2m'], current['wind_speed_10m']
        )
        dew_point = calculate_dew_point(current['temperature_2m'], current['relative_humidity_2m'])

        # Current metrics with text labels
        cols = st.columns(4)
        metrics = [
            (f"{current['temperature_2m']:.1f}°C", "Temperature", "TEMP"),
            (f"{apparent_temp:.1f}°C", "Feels Like", "FEEL"),
            (f"{current['relative_humidity_2m']:.0f}%", "Humidity", "HUM"),
            (f"{dew_point:.1f}°C", "Dew Point", "DEW")
        ]
        for col, (val, label, short) in zip(cols, metrics):
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-icon'>{short}</div>
                    <div class='metric-value'>{val}</div>
                    <div>{label}</div>
                </div>
                """, unsafe_allow_html=True)

        cols = st.columns(4)
        metrics2 = [
            (f"{current['wind_speed_10m']:.1f} km/h", "Wind Speed", "WIND"),
            (f"{current['pressure_msl']:.0f} hPa", "Pressure", "PRES"),
            (f"{current['precipitation']:.1f} mm", "Precipitation", "RAIN"),
            (f"{current['cloud_cover']:.0f}%", "Cloud Cover", "CLD")
        ]
        for col, (val, label, short) in zip(cols, metrics2):
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-icon'>{short}</div>
                    <div class='metric-value'>{val}</div>
                    <div>{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'>24-Hour Forecast</div>", unsafe_allow_html=True)
        for i in range(0, 12, 4):
            cols = st.columns(4)
            for j, col in enumerate(cols):
                if i + j < len(fc_24h):
                    row = fc_24h.iloc[i + j]
                    with col:
                        st.markdown(f"""
                        <div class='forecast-card'>
                            <div style='font-weight:600;color:#1e40af'>{row['time'].strftime('%H:%M')}</div>
                            <div style='text-align:center;margin:10px 0;font-size:1.2rem;font-weight:bold;color:#4F87FF'>
                                {get_weather_display(row['weather_condition'])}
                            </div>
                            <div class='condition-badge'>{row['weather_condition']}</div>
                            <div style='text-align:center'>
                                <strong>{row['temperature_2m']:.1f}°C</strong><br>
                                {row['precipitation']:.1f}mm • {row['wind_speed_10m']:.0f}km/h
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    # === DETAILED FORECAST ===
    elif page == "Detailed Forecast":
        st.markdown("<div class='section-header'>Detailed Weather Forecast</div>", unsafe_allow_html=True)
        forecast_options = {"24 Hours": 24, "48 Hours": 48, "3 Days (72 Hours)": 72, "5 Days (120 Hours)": 120, "7 Days (168 Hours)": 168}
        selected = st.selectbox("Select Forecast Duration", list(forecast_options.keys()), index=2)
        hours = forecast_options[selected]

        if st.button("Generate Forecast", type="primary", use_container_width=True):
            with st.spinner(f"Generating {selected} forecast..."):
                fc = predictor.generate_forecast(hist, hours=hours)
                st.session_state.detailed_forecast = fc
            st.success("Forecast generated successfully!")

        if st.session_state.detailed_forecast is not None:
            df = st.session_state.detailed_forecast.copy()

            tab1, tab2 = st.tabs(["Comprehensive View", "Data Table"])

            with tab1:
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    subplot_titles=("Temperature & Apparent Temperature", "Precipitation & Humidity", "Wind Speed & Cloud Cover"))
                
                # Check if apparent_temperature exists before plotting
                if 'apparent_temperature' in df.columns:
                    fig.add_trace(go.Scatter(x=df['time'], y=df['apparent_temperature'], name="Feels Like", line=dict(color='#ff4444', width=2, dash='dash')), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=df['time'], y=df['temperature_2m'], name="Temperature", line=dict(color='#ff7f0e', width=3)), row=1, col=1)
                fig.add_trace(go.Bar(x=df['time'], y=df['precipitation'], name="Precipitation", marker_color='#1f77b4', opacity=0.7), row=2, col=1)
                fig.add_trace(go.Scatter(x=df['time'], y=df['relative_humidity_2m'], name="Humidity", line=dict(color='#d62728', width=2)), row=2, col=1)
                fig.add_trace(go.Scatter(x=df['time'], y=df['wind_speed_10m'], name="Wind Speed", line=dict(color='#2ca02c', width=3)), row=3, col=1)
                fig.add_trace(go.Scatter(x=df['time'], y=df['cloud_cover'], name="Cloud Cover", line=dict(color='#9467bd', width=2)), row=3, col=1)
                
                fig.update_layout(height=900, title_text=f"Detailed Weather Forecast - {selected}", showlegend=True)
                fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
                fig.update_yaxes(title_text="Precipitation (mm)", row=2, col=1)
                fig.update_yaxes(title_text="Wind Speed (km/h)", row=3, col=1)
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                display_df = df.copy()
                display_df['time'] = display_df['time'].dt.strftime('%Y-%m-%d %H:%M')
                display_df = display_df.round(2)
                st.dataframe(display_df, use_container_width=True, height=600)

    # === HISTORICAL DATA - ALL PARAMETERS TOGETHER ===
    elif page == "Historical Data":
        st.markdown("<div class='section-header'>Historical Weather Data Analysis</div>", unsafe_allow_html=True)
        
        # Time period selection
        col1, col2 = st.columns(2)
        with col1:
            period = st.selectbox("Select Time Period", 
                                ["Last 7 days", "Last 30 days", "Last 90 days", "Last 365 days", "All available data"])
        with col2:
            st.markdown("### All Weather Parameters")
            st.info("Displaying comprehensive analysis for all weather parameters")
        
        # Filter data based on selection
        if period == "Last 7 days":
            data = hist.tail(168)
        elif period == "Last 30 days":
            data = hist.tail(720)
        elif period == "Last 90 days":
            data = hist.tail(2160)
        elif period == "Last 365 days":
            data = hist.tail(8760)
        else:
            data = hist
        
        # Overview Statistics
        st.markdown("###Overview Statistics")
        stats_cols = st.columns(6)
        parameters = ['temperature_2m', 'precipitation', 'wind_speed_10m', 
                     'relative_humidity_2m', 'cloud_cover', 'pressure_msl']
        param_names = ['Temperature (°C)', 'Precipitation (mm)', 'Wind Speed (km/h)', 
                      'Humidity (%)', 'Cloud Cover (%)', 'Pressure (hPa)']
        
        for i, (param, name) in enumerate(zip(parameters, param_names)):
            with stats_cols[i]:
                st.metric(
                    label=name,
                    value=f"{data[param].mean():.1f}",
                    delta=f"Range: {data[param].max()-data[param].min():.1f}"
                )
        
        # COMPREHENSIVE MULTI-PARAMETER CHART
        st.markdown("###  Multi-Parameter Time Series")
        
        fig_comprehensive = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Temperature & Feels Like', 'Precipitation',
                'Humidity & Cloud Cover', 'Wind Speed',
                'Pressure', 'All Parameters Overview'
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # Temperature and Apparent Temperature
        fig_comprehensive.add_trace(
            go.Scatter(x=data['time'], y=data['temperature_2m'], name='Temperature', 
                      line=dict(color='#ff7f0e', width=2)),
            row=1, col=1
        )
        if 'apparent_temperature' in data.columns:
            fig_comprehensive.add_trace(
                go.Scatter(x=data['time'], y=data['apparent_temperature'], name='Feels Like',
                          line=dict(color='#ff4444', width=2, dash='dash')),
                row=1, col=1
            )
        
        # Precipitation
        fig_comprehensive.add_trace(
            go.Bar(x=data['time'], y=data['precipitation'], name='Precipitation',
                  marker_color='#1f77b4', opacity=0.7),
            row=1, col=2
        )
        
        # Humidity and Cloud Cover
        fig_comprehensive.add_trace(
            go.Scatter(x=data['time'], y=data['relative_humidity_2m'], name='Humidity',
                      line=dict(color='#d62728', width=2)),
            row=2, col=1
        )
        fig_comprehensive.add_trace(
            go.Scatter(x=data['time'], y=data['cloud_cover'], name='Cloud Cover',
                      line=dict(color='#9467bd', width=2)),
            row=2, col=1
        )
        
        # Wind Speed
        fig_comprehensive.add_trace(
            go.Scatter(x=data['time'], y=data['wind_speed_10m'], name='Wind Speed',
                      line=dict(color='#2ca02c', width=2)),
            row=2, col=2
        )
        
        # Pressure
        fig_comprehensive.add_trace(
            go.Scatter(x=data['time'], y=data['pressure_msl'], name='Pressure',
                      line=dict(color='#8c564b', width=2)),
            row=3, col=1
        )
        
        # All parameters normalized overview
        for param, color in zip(parameters, ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b']):
            normalized = (data[param] - data[param].min()) / (data[param].max() - data[param].min())
            fig_comprehensive.add_trace(
                go.Scatter(x=data['time'], y=normalized, name=param.replace('_', ' ').title(),
                          line=dict(color=color, width=1), showlegend=False),
                row=3, col=2
            )
        
        fig_comprehensive.update_layout(height=1200, title_text=f"Comprehensive Weather Analysis - {period}", showlegend=True)
        st.plotly_chart(fig_comprehensive, use_container_width=True)
        
        # DISTRIBUTION ANALYSIS
        st.markdown("###  Parameter Distributions")
        
        dist_cols = st.columns(3)
        dist_params = [('temperature_2m', 'Temperature Distribution'), 
                      ('precipitation', 'Precipitation Distribution'),
                      ('wind_speed_10m', 'Wind Speed Distribution')]
        
        for i, (param, title) in enumerate(dist_params):
            with dist_cols[i]:
                fig_dist = px.histogram(data, x=param, title=title, 
                                      color_discrete_sequence=['#4F87FF'])
                fig_dist.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_dist, use_container_width=True)
        
        dist_cols2 = st.columns(3)
        dist_params2 = [('relative_humidity_2m', 'Humidity Distribution'),
                       ('cloud_cover', 'Cloud Cover Distribution'),
                       ('pressure_msl', 'Pressure Distribution')]
        
        for i, (param, title) in enumerate(dist_params2):
            with dist_cols2[i]:
                fig_dist = px.histogram(data, x=param, title=title,
                                      color_discrete_sequence=['#6AC9FF'])
                fig_dist.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_dist, use_container_width=True)
        
      
        
        # RAW DATA TABLE
        st.markdown("###  Raw Data Table")
        
        display_data = data.copy()
        display_data['time'] = display_data['time'].dt.strftime('%Y-%m-%d %H:%M')
        display_data = display_data.round(2)
        
        st.dataframe(display_data, use_container_width=True, height=400)
        
        # Export functionality
        csv = display_data.to_csv(index=False)
        st.download_button(
            label="Download Historical Data (CSV)",
            data=csv,
            file_name=f"kathmandu_historical_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # === ABOUT PAGE ===
    elif page == "About":
        st.markdown("""
        ## Kathmandu AI Weather System
        
        ### Advanced Features:
        
        **AI-Powered Forecasting**
        - Ensemble Random Forest models for multi-parameter prediction
        - Recursive multi-step forecasting with dynamic feature engineering
        - Physics-based fallback systems for error resilience
        
        **Enhanced Visualizations**
        - Interactive charts with multiple parameters
        - Historical trend analysis with statistical insights
        - Real-time data processing and display
        
        **Location-Specific**
        - Optimized for Kathmandu valley climate patterns
        - Accurate seasonal and diurnal variations
        - Mountain weather pattern recognition
        
        **Performance Optimizations**
        - Smart caching for fast data retrieval
        - Efficient data streaming and processing
        - Real-time API integration
        
        **Advanced Metrics**
        - Apparent temperature calculations
        - Dew point and humidity analysis
        - Comprehensive weather parameter tracking
        
        ### Data Sources:
        - Open-Meteo API for real-time and historical data
        - Machine learning models trained on local climate patterns
        - Realistic fallback data generation
        
        *Built with Streamlit, Plotly, and Scikit-learn*
        
        ### Technical Details:
        - **Forecast Method:** Recursive multi-step prediction
        - **Data Resolution:** Hourly updates
        - **Model Type:** Random Forest Ensemble
        - **Coverage:** Kathmandu Metropolitan Area
        """)

if __name__ == "__main__":
    main()