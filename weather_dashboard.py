# weather_dashboard.py 
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# PAGE CONFIG
st.set_page_config(page_title="Kathmandu Weather", page_icon="logo.png", layout="wide", initial_sidebar_state="expanded")

#CSS
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); font-family: 'Segoe UI', sans-serif; color: white;}
    
    .header-title {
        font-size: 3.5rem; font-weight: 900; text-align: center;
        background: linear-gradient(90deg, #FFD700, #FFA500, #FF6B6B);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        padding: 2rem 0 1rem; margin-bottom: 0;
    }
    .subtitle {text-align: center; font-size: 1.3rem; color: #e0e7ff; margin-bottom: 2rem;}
    .section-header {
        font-size: 2rem; font-weight: 700; color: #FFD700;
        border-bottom: 3px solid #FFA500; padding-bottom: 10px; margin: 2.5rem 0 1.5rem;
    }

    .metric-card, .forecast-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        padding: 1.8rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
        height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s;
    }
    .metric-value, .forecast-temp {
        font-weight: 800;
        background: linear-gradient(90deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-value {font-size: 2.8rem; margin-bottom: 0.5rem;}
    .forecast-temp {font-size: 2.2rem; margin: 0.5rem 0;}

    .condition-badge {
        background: linear-gradient(90deg, #FF6B6B, #FF8E53);
        color: white;
        padding: 0.7rem 1.2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.9rem;
    }

    .stButton button {
        width: 100%;
        background: linear-gradient(90deg, #FF6B6B, #FF88E5);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 10px;
        font-weight: 600;
    }
    .stButton button:hover {
        background: linear-gradient(90deg, #FF8E53, #FF6B6B);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    /* DROPDOWNS: NO TYPING, 100% CLICKABLE */
    .stSelectbox > div > div > div > input {
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        position: absolute !important;
        pointer-events: none !important;
    }
    .stSelectbox [data-baseweb="select"] > div {
        pointer-events: auto !important;
        cursor: pointer !important;
    }
    .stSelectbox > div > div > div > div {
        color: white !important;
        font-weight: 600 !important;
    }
    .stSelectbox svg {
        color: #FFD700 !important;
    }
</style>
""", unsafe_allow_html=True)


with st.spinner("Loading model & historical data"):
    @st.cache_resource(show_spinner=False)
    def load_model():
        return joblib.load("kathmandu_trend_model.pkl")
    
    @st.cache_data(ttl=86400, show_spinner=False)
    def load_historical():
        df = pd.read_csv("open-meteo-27.75N85.50E1293m.csv")
        df['time'] = pd.to_datetime(df['time'])
        return df.sort_values('time').reset_index(drop=True)

    model_data = load_model()
    historical_df = load_historical()

models = model_data['models']
feature_cols = model_data['feature_cols']

# PREDICTOR
class TrendPredictor:
    def predict(self, dt):
        f = {
            'hour_sin': np.sin(2 * np.pi * dt.hour / 24),
            'hour_cos': np.cos(2 * np.pi * dt.hour / 24),
            'month_sin': np.sin(2 * np.pi * dt.month / 12),
            'month_cos': np.cos(2 * np.pi * dt.month / 12),
            'doy_sin': np.sin(2 * np.pi * dt.timetuple().tm_yday / 365),
            'doy_cos': np.cos(2 * np.pi * dt.timetuple().tm_yday / 365),
            'is_monsoon': int(dt.month in [6,7,8,9]),
            'year_progress': dt.timetuple().tm_yday / 365,
            'days_since_2020': (dt - pd.Timestamp("2020-01-01")).days
        }
        X = np.array([[f[col] for col in feature_cols]])
        pred = {'time': dt}
        for target, info in models.items():
            pred[target] = max(0, info['model'].predict(info['scaler'].transform(X))[0])
        pred['weather_condition'] = self.get_condition(pred)
        pred['apparent_temperature'] = pred['temperature_2m'] + (pred['relative_humidity_2m'] - 50) * 0.1
        return pred

    def get_condition(self, p):
        if p['precipitation'] > 10: return "Heavy Rain"
        if p['precipitation'] > 4: return "Rain"
        if p['precipitation'] > 0.5: return "Light Rain"
        if p['cloud_cover'] > 80: return "Overcast"
        if p['cloud_cover'] > 60: return "Mostly Cloudy"
        if p['cloud_cover'] > 30: return "Partly Cloudy"
        return "Clear Sky"

predictor = TrendPredictor()

with st.spinner("Generating fresh 7-day forecast..."):
    @st.cache_data(ttl=1800, show_spinner=False)
    def get_forecast(hours: int):
        start = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        times = pd.date_range(start, periods=hours, freq='H')
        return [predictor.predict(t) for t in times], times

# CURRENT WEATHER
@st.cache_data(ttl=300, show_spinner=False)  # 5 min cache
def get_current():
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": 27.7172, "longitude": 85.3240,
            "current": "temperature_2m,relative_humidity_2m,precipitation,cloud_cover,wind_speed_10m",
            "timezone": "Asia/Kathmandu"
        }, timeout=10)
        return r.json()['current']
    except:
        return None

current = get_current()

# HEADER
st.markdown("<div class='header-title'>Kathmandu Weather Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>ML Based • 7-Day Forecast</div>", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("### Kathmandu Weather Prediction")
    st.markdown("**Location:** Kathmandu Valley • **Elevation:** ~1,400m")
    st.markdown(f"**Updated:** {datetime.now().strftime('%d %b %Y, %H:%M')}")
    page = st.radio("Navigate", ["Dashboard", "Detailed Forecast", "Historical Trends", "About"], label_visibility="collapsed")
    if st.button("Refresh All", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Refreshed!")
        st.rerun()

# DASHBOARD
if page == "Dashboard":
    temp = current['temperature_2m'] if current else 19.0
    hum = current['relative_humidity_2m'] if current else 72
    rain = current['precipitation'] if current else 0.0
    dew = temp - ((100 - hum)/5)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f"<div class='metric-card'><div class='metric-value'>{temp:.1f}°C</div><div>Temperature</div></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric-card'><div class='metric-value'>{dew:.1f}°C</div><div>Dew Point</div></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric-card'><div class='metric-value'>{hum}%</div><div>Humidity</div></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='metric-card'><div class='metric-value'>{rain:.1f}mm</div><div>Rain</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>12hr Forecast</div>", unsafe_allow_html=True)
    forecast_12h, _ = get_forecast(12)
    for i in range(0, 12, 4):
        cols = st.columns(4)
        for j, col in enumerate(cols):
            if i + j < len(forecast_12h):
                row = forecast_12h[i + j]
                color = "#FF4757" if row['precipitation'] > 5 else "#FF6B6B" if row['precipitation'] > 1 else "#FFD93D" if row['precipitation'] > 0.2 else "#8CC8FF"
                with col:
                    st.markdown(f"""
                    <div class='forecast-card' style='--border-color: {color}'>
                        <div class='forecast-time'>{row['time'].strftime('%H:%M')}</div>
                        <div class='condition-badge'>{row['weather_condition']}</div>
                        <div class='forecast-temp'>{row['temperature_2m']:.1f}°C</div>
                        <div class='forecast-details'>Rain {row['precipitation']:.1f}mm • Wind {row['wind_speed_10m']:.0f} km/h</div>
                    </div>
                    """, unsafe_allow_html=True)
        if i < 8: st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>7-Day Temperature Trend</div>", unsafe_allow_html=True)
    forecast_7d, _ = get_forecast(168)
    df_7d = pd.DataFrame(forecast_7d)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_7d['time'], y=df_7d['temperature_2m'], name='Temperature', line=dict(color='#FFD700', width=4)))
    fig.add_trace(go.Scatter(x=df_7d['time'], y=df_7d['apparent_temperature'], name='Feels Like', line=dict(color='#FF6B6B', dash='dot')))
    fig.update_layout(height=480, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.1)')
    st.plotly_chart(fig, use_container_width=True)

# DETAILED FORECAST
elif page == "Detailed Forecast":
    st.markdown("<div class='section-header'>Detailed Forecast</div>", unsafe_allow_html=True)
    hours = st.selectbox("Select Duration", [72, 120, 168], format_func=lambda x: f"{x//24} Days", key="detail_hours")
    forecast, _ = get_forecast(hours)
    df = pd.DataFrame(forecast)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Temperature & Feels Like", "Precipitation & Humidity", "Wind & Cloud Cover"))
    fig.add_trace(go.Scatter(x=df['time'], y=df['temperature_2m'], name='Temp', line=dict(color='#FFD700')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['apparent_temperature'], name='Feels', line=dict(color='#FF6B6B', dash='dash')), row=1, col=1)
    fig.add_trace(go.Bar(x=df['time'], y=df['precipitation'], name='Rain', marker_color='#4ECDC4'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['relative_humidity_2m'], name='Humidity', line=dict(color='#95E1D3')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['wind_speed_10m'], name='Wind', line=dict(color='#FF8E53')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['cloud_cover'], name='Cloud', line=dict(color='#A0A0A0')), row=3, col=1)
    fig.update_layout(height=800, template="plotly_dark", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df.round(2)[['time','temperature_2m','precipitation','relative_humidity_2m','wind_speed_10m','cloud_cover','weather_condition']], use_container_width=True)

# HISTORICAL & ABOUT
elif page == "Historical Trends":
    st.markdown("<div class='section-header'>Complete Historical Weather Trends</div>", unsafe_allow_html=True)
    period = st.selectbox("Select Period", ["Last 30 Days", "Last 90 Days", "Last Year", "All Data"], index=0)
    data = historical_df.tail(720) if period == "Last 30 Days" else \
           historical_df.tail(2160) if period == "Last 90 Days" else \
           historical_df[historical_df['time'] >= historical_df['time'].max() - timedelta(days=365)] if period == "Last Year" else historical_df.copy()

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Temperature (°C)", "Precipitation (mm)", "Humidity & Cloud Cover (%)", "Wind Speed (km/h)"))
    fig.add_trace(go.Scatter(x=data['time'], y=data['temperature_2m'], name='Temp', line=dict(color='#FFD700')), row=1, col=1)
    fig.add_trace(go.Bar(x=data['time'], y=data['precipitation'], name='Rain', marker_color="#1B89F7"), row=2, col=1)
    fig.add_trace(go.Scatter(x=data['time'], y=data['relative_humidity_2m'], name='Humidity', line=dict(color='#95E1D3')), row=3, col=1)
    fig.add_trace(go.Scatter(x=data['time'], y=data['cloud_cover'], name='Cloud', line=dict(color='#A0A0A0')), row=3, col=1)
    fig.add_trace(go.Scatter(x=data['time'], y=data['wind_speed_10m'], name='Wind', line=dict(color='#FF8E53')), row=4, col=1)
    fig.update_layout(height=900, template="plotly_dark", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Historical Data Table")
    st.dataframe(data.round(2).reset_index(drop=True), use_container_width=True, height=500, hide_index=True)

elif page == "About":
    st.markdown("# Kathmandu Weather Prediction")
    st.markdown("**Kathmandu Weather Prediction** is an intelligent weather forecasting system that combines historical data analysis" \
    " with machine learning to provide accurate weather predictions for the Kathmandu Valley.")

st.caption(f"Data: {historical_df['time'].min().date()} to {historical_df['time'].max().date()}")