# weather_dashboard_enhanced.py
import pandas as pd
import streamlit as st
import joblib
from datetime import datetime, timedelta
import numpy as np
import plotly.express as px
import os

# -------------------- SETUP --------------------
st.set_page_config(
    page_title="Advanced Weather Forecast Dashboard",
    layout="wide",
    page_icon="🌤️",
    initial_sidebar_state="expanded"
)

# -------------------- STYLES --------------------
def load_css():
    st.markdown("""
        <style>
        .main { background-color: #0f172a; color: white; }
        .stButton>button {
            background-color: #3b82f6; color: white;
            border: none; border-radius: 6px;
            padding: 0.5rem 1rem;
        }
        .hour-card {
            background-color: #1e293b;
            border-radius: 10px;
            padding: 12px;
            text-align: center;
            margin-bottom: 12px;
        }
        .current-weather {
            background: linear-gradient(135deg, #3b82f6, #1e40af);
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 24px;
        }
        .disclaimer {
            background-color: #fef3c7;
            color: #92400e;
            padding: 12px;
            border-radius: 10px;
            margin: 12px 0;
        }
        </style>
    """, unsafe_allow_html=True)

load_css()

# -------------------- DATA LOADING --------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("optimized_weather_model.pkl")
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None

@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv("open-meteo-27.73N85.25E1293m.csv")
        df['time'] = pd.to_datetime(df['time'])
        df['hour'] = df['time'].dt.hour
        df['month'] = df['time'].dt.month
        df['day_of_week'] = df['time'].dt.weekday

        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Weather code mapping
        weather_map = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Fog", 51: "Drizzle", 53: "Drizzle", 55: "Drizzle",
            61: "Rain", 63: "Rain", 65: "Rain", 71: "Snow", 73: "Snow", 75: "Snow",
            77: "Snow grains", 80: "Rain showers", 81: "Rain showers", 82: "Rain showers",
            85: "Snow showers", 86: "Snow showers", 95: "Thunderstorm",
            96: "Thunderstorm with hail", 99: "Thunderstorm with hail"
        }
        df['weather_description'] = df['weather_code'].map(weather_map).fillna("Unknown")
        return df
    except Exception as e:
        st.error(f"❌ Data loading failed: {str(e)}")
        return pd.DataFrame()

# -------------------- EXTERNAL DATA FETCHING --------------------
@st.cache_data(ttl=3600)
def fetch_weather_comparison():
    today = datetime.now().date()
    dates = [today + timedelta(days=i) for i in range(3)]
    return pd.DataFrame({
        'date': dates,
        'temp_high': [25.3, 26.1, 24.8],
        'temp_low': [18.2, 19.5, 17.8],
        'condition': ['Partly Cloudy', 'Scattered Thunderstorms', 'Mostly Sunny'],
        'precipitation_chance': [10, 60, 5]
    })

# -------------------- UTILITIES --------------------
def get_weather_icon_and_text(code):
    mapping = {
        0: ("wb_sunny", "Clear", "#facc15"),
        2: ("cloud", "Partly Cloudy", "#f1f5f9"),
        3: ("cloud_queue", "Overcast", "#cbd5e1"),
        61: ("cloud_rain", "Rain", "#3b82f6"),
        81: ("storm", "Rain Showers", "#f87171"),
        95: ("storm", "Thunderstorm", "#f87171"),
        99: ("storm", "Heavy Thunderstorm", "#dc2626")
    }
    return mapping.get(int(code), ("help_outline", "Unknown", "#f1f5f9"))

# -------------------- FORECAST --------------------
def generate_hourly_forecast(date, df, model):
    forecasts, median_values = [], df.median(numeric_only=True)

    for hour in range(24):
        input_data = {
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'month_sin': np.sin(2 * np.pi * date.month / 12),
            'month_cos': np.cos(2 * np.pi * date.month / 12),
            'hour': hour, 'month': date.month, 'day_of_week': date.weekday()
        }

        similar = df[(df['hour'] == hour) & (df['month'] == date.month)]
        if not similar.empty:
            input_data.update({
                'weather_code': int(similar['weather_code'].mode()[0]),
                'wind_direction_100m': similar['wind_direction_100m'].median(),
                'wind_speed_10m (km/h)': similar['wind_speed_10m (km/h)'].median(),
                'relative_humidity_2m': similar['relative_humidity_2m'].median()
            })
        else:
            input_data.update({
                'weather_code': 0,
                'wind_direction_100m': median_values.get('wind_direction_100m', 0),
                'wind_speed_10m (km/h)': median_values.get('wind_speed_10m (km/h)', 0),
                'relative_humidity_2m': median_values.get('relative_humidity_2m', 50)
            })

        input_df = pd.DataFrame([input_data])
        for f in set(model.feature_names_in_) - set(input_df.columns):
            input_df[f] = 0

        pred = model.predict(input_df[model.feature_names_in_])
        temp = round(float(pred[0, 0]) if pred.ndim > 1 else float(pred[0]), 1)

        forecasts.append({
            'hour': hour,
            'time': f"{hour:02d}:00",
            'temperature': temp,
            'weather_code': input_data['weather_code'],
            'humidity': input_data['relative_humidity_2m'],
            'wind_speed': input_data['wind_speed_10m (km/h)'],
            'wind_direction': input_data['wind_direction_100m']
        })
    return forecasts

# -------------------- DASHBOARD --------------------
def prediction_page(df, model):
    st.title("Kathmandu Weather Forecast")
    st.markdown("""
    <div class="disclaimer"><strong>Note:</strong> Forecast is based on historical data patterns and may differ 
    from professional services that use satellites, radar, and atmospheric models.</div>
    """, unsafe_allow_html=True)

    selected_date = st.date_input("Select date", datetime.now().date())
    if model is None or df.empty:
        st.error("⚠️ Forecast unavailable. Missing model or data.")
        return

    forecast = generate_hourly_forecast(selected_date, df, model)
    current = forecast[min(datetime.now().hour, 23)]
    icon, text, color = get_weather_icon_and_text(current['weather_code'])

    st.markdown(f"""
    <div class="current-weather">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <div style="display:flex;align-items:center;gap:16px">
                <span style="font-size:3rem;font-weight:600">{current['temperature']}°C</span>
                <div style="text-align:center">
                    <span class="material-icons" style="font-size:3rem;color:{color}">{icon}</span>
                    <div style="font-size:1rem;color:#e2e8f0">{text}</div>
                </div>
            </div>
            <div style="display:flex;flex-direction:column;align-items:flex-end;gap:8px;color:#e2e8f0">
                <div><span class="material-icons">schedule</span> {current['time']}</div>
                <div><span class="material-icons">air</span> {current['wind_speed']:.1f} km/h</div>
                <div><span class="material-icons">water_drop</span> {int(current['humidity'])}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hourly forecast cards
    st.subheader("Hourly Forecast (Next 8 Hours)")
    cols = st.columns(8)
    for i, hour in enumerate(forecast[:8]):
        icon, text, color = get_weather_icon_and_text(hour['weather_code'])
        with cols[i % 8]:
            st.markdown(f"""
            <div class="hour-card">
                <div>{hour['time']}</div>
                <span class="material-icons" style="color:{color}">{icon}</span>
                <div>{hour['temperature']}°C</div>
                <div style="font-size:0.8rem;color:#94a3b8">{text}</div>
            </div>
            """, unsafe_allow_html=True)

    # Chart
    forecast_df = pd.DataFrame(forecast)
    fig = px.line(forecast_df, x='time', y='temperature',
                  labels={'time': 'Hour', 'temperature': '°C'})
    fig.update_layout(plot_bgcolor='#0f172a', paper_bgcolor='#0f172a', font_color='white')
    st.plotly_chart(fig, use_container_width=True)

# -------------------- MAIN --------------------
def main():
    df, model = load_historical_data(), load_model()
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("", ["Forecast"])
    if page == "Forecast":
        prediction_page(df, model)

if __name__ == "__main__":
    main()
