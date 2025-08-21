#weather_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import altair as alt
import os

# -------------------- PAGE SETUP --------------------
st.set_page_config(
    page_title="Weather Dashboard",
    layout="wide",
    page_icon="🌤️"
)

# Load CSS
if os.path.exists('styles.css'):
    with open('styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Custom header
st.markdown("""
<div class="header">
    <i class="fas fa-cloud-sun header-icon"></i>
    <h1 class="header-title">Hourly Weather Dashboard</h1>
</div>
""", unsafe_allow_html=True)

# -------------------- LOAD DATA & MODEL --------------------
csv_file = "open-meteo-27.75N85.50E1293mNew.csv"
model_file = "weather_prediction_hourly_model.pkl"

df_hist = pd.DataFrame()
model_data = {}

if not os.path.exists(csv_file):
    st.error(f"Error: '{csv_file}' not found.")
    st.stop()

try:
    df_hist = pd.read_csv(csv_file)
    df_hist['time'] = pd.to_datetime(df_hist['time'], errors='coerce')
    df_hist.dropna(subset=['time'], inplace=True)
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# Feature engineering
feature_cols = []
if not df_hist.empty:
    df_hist['hour'] = df_hist['time'].dt.hour.astype(int)
    df_hist['day_of_week'] = df_hist['time'].dt.dayofweek
    df_hist['month'] = df_hist['time'].dt.month
    df_hist['year'] = df_hist['time'].dt.year
    df_hist['is_weekend'] = df_hist['day_of_week'].isin([5,6]).astype(int)

    target_columns_potential = [
        'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation',
        'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m'
    ]
    numerical_cols = df_hist.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numerical_cols if col not in target_columns_potential]

    for col in feature_cols:
        df_hist[col].fillna(df_hist[col].median(), inplace=True)
else:
    st.info("Historical data is empty or could not be loaded.")

# Load model
if not os.path.exists(model_file):
    st.error(f"Error: '{model_file}' not found.")
    st.stop()
try:
    model_data = joblib.load(model_file)
    if not all(k in model_data for k in ['target_columns', 'models', 'feature_columns']):
        st.error("Model structure unexpected. Missing keys.")
        st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -------------------- DATE SELECTION --------------------
min_date = datetime.now().date()
max_date = min_date + timedelta(days=7)
selected_date = st.date_input("Select Date", value=min_date, min_value=min_date, max_value=max_date)

# Metric dictionaries
metric_units = {
    'temperature_2m': '°C', 'relative_humidity_2m': '%', 'dew_point_2m': '°C',
    'precipitation': 'mm', 'wind_speed_10m': 'km/h', 'wind_direction_10m': '°',
    'wind_gusts_10m': 'km/h'
}
metric_display_names = {
    'temperature_2m': 'Temperature', 'relative_humidity_2m': 'Humidity', 'dew_point_2m': 'Dew Point',
    'precipitation': 'Precipitation', 'wind_speed_10m': 'Wind Speed', 'wind_direction_10m': 'Wind Direction',
    'wind_gusts_10m': 'Wind Gusts'
}
metric_icons = {
    'temperature_2m': 'fas fa-thermometer-half', 'relative_humidity_2m': 'fas fa-tint', 
    'dew_point_2m': 'fas fa-cloud-rain', 'precipitation': 'fas fa-cloud-showers-heavy',
    'wind_speed_10m': 'fas fa-wind', 'wind_direction_10m': 'fas fa-compass',
    'wind_gusts_10m': 'fas fa-wind'
}

# -------------------- FUTURE FEATURES --------------------
def create_future_features_for_date(date, hours=24):
    future_times = [datetime.combine(date, datetime.min.time()) + timedelta(hours=i) for i in range(hours)]
    df_future = pd.DataFrame({'time': future_times})
    df_future['hour'] = df_future['time'].dt.hour.astype(int)
    df_future['day_of_week'] = df_future['time'].dt.dayofweek
    df_future['month'] = df_future['time'].dt.month
    df_future['year'] = df_future['time'].dt.year
    df_future['is_weekend'] = df_future['day_of_week'].isin([5, 6]).astype(int)

    for col in model_data['feature_columns']:
        if col in df_hist.columns and not df_hist[col].empty:
            df_future[col] = df_hist[col].median()
        else:
            df_future[col] = 0
    return df_future

def predict_weather(df_future, model_data):
    predictions = pd.DataFrame({'time': df_future['time'], 'hour': df_future['hour']})
    for col in model_data['feature_columns']:
        if col not in df_future.columns:
            df_future[col] = 0

    for target in model_data['target_columns']:
        if target in model_data['models']:
            predictions[target] = model_data['models'][target].predict(df_future[model_data['feature_columns']])
        else:
            predictions[target] = 0
    return predictions

df_future_features = create_future_features_for_date(selected_date)
predictions = predict_weather(df_future_features, model_data)

# -------------------- TABS --------------------
tabs = st.tabs([
    "Current Weather", "Hourly Forecast", "Temperature", "Humidity", "Wind", "Precipitation"
])

# -------------------- CURRENT WEATHER --------------------
with tabs[0]:
    st.markdown("<h2>Current Hour Weather</h2>", unsafe_allow_html=True)
    current_hour_dt = datetime.now()
    current_hour_value = current_hour_dt.hour
    current_weather = predictions[
        (predictions['time'].dt.date == selected_date) &
        (predictions['hour'] == current_hour_value)
    ]
    if not current_weather.empty:
        row = current_weather.iloc[0]
        st.markdown('<div class="current-weather-grid">', unsafe_allow_html=True)
        for metric in ['temperature_2m','relative_humidity_2m','dew_point_2m','precipitation','wind_speed_10m','wind_gusts_10m']:
            st.markdown(f"""
            <div class="weather-metric">
                <i class="{metric_icons[metric]} metric-icon"></i>
                <span class="metric-value">{row.get(metric, 0):.1f}</span>
                <span class="metric-unit">{metric_units.get(metric,'')}</span>
                <span class="metric-label">{metric_display_names[metric]}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.write("No prediction available for the current hour.")

# -------------------- HOURLY FORECAST --------------------
with tabs[1]:
    st.markdown(f"<h2>Hourly Weather for {selected_date.strftime('%Y-%m-%d')}</h2>", unsafe_allow_html=True)
    st.markdown('<div class="hourly-cards-grid">', unsafe_allow_html=True)

    for idx, row in predictions.iterrows():
        st.markdown(f"""
        <div class="hourly-card">
            <div class="hourly-time">{int(row['hour']):02d}:00</div>
            <div class="hourly-metrics">
                <div class="metric"><i class="{metric_icons['temperature_2m']}"></i> <span>{row.get('temperature_2m',0):.1f} {metric_units['temperature_2m']}</span></div>
                <div class="metric"><i class="{metric_icons['relative_humidity_2m']}"></i> <span>{row.get('relative_humidity_2m',0):.1f} {metric_units['relative_humidity_2m']}</span></div>
                <div class="metric"><i class="{metric_icons['dew_point_2m']}"></i> <span>{row.get('dew_point_2m',0):.1f} {metric_units['dew_point_2m']}</span></div>
                <div class="metric"><i class="{metric_icons['precipitation']}"></i> <span>{row.get('precipitation',0):.2f} {metric_units['precipitation']}</span></div>
                <div class="metric"><i class="{metric_icons['wind_speed_10m']}"></i> <span>{row.get('wind_speed_10m',0):.1f} {metric_units['wind_speed_10m']}</span></div>
                <div class="metric"><i class="{metric_icons['wind_gusts_10m']}"></i> <span>{row.get('wind_gusts_10m',0):.1f} {metric_units['wind_gusts_10m']}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- TEMPERATURE CHART --------------------
with tabs[2]:
    st.markdown("<h2>Temperature Chart</h2>", unsafe_allow_html=True)
    if 'temperature_2m' in predictions.columns:
        chart = alt.Chart(predictions).mark_line(point=True).encode(
            x=alt.X('hour:O', title='Hour'),
            y=alt.Y('temperature_2m', title='Temperature (°C)'),
            tooltip=[alt.Tooltip('hour', title='Hour'), alt.Tooltip('temperature_2m', title='Temp', format='.1f')]
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

# -------------------- HUMIDITY CHART --------------------
with tabs[3]:
    st.markdown("<h2>Humidity Chart</h2>", unsafe_allow_html=True)
    if 'relative_humidity_2m' in predictions.columns:
        chart = alt.Chart(predictions).mark_line(point=True, color='blue').encode(
            x=alt.X('hour:O', title='Hour'),
            y=alt.Y('relative_humidity_2m', title='Humidity (%)'),
            tooltip=[alt.Tooltip('hour', title='Hour'), alt.Tooltip('relative_humidity_2m', title='Humidity', format='.1f')]
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

# -------------------- WIND CHART --------------------
with tabs[4]:
    st.markdown("<h2>Wind Chart</h2>", unsafe_allow_html=True)
    if all(c in predictions.columns for c in ['wind_speed_10m','wind_gusts_10m']):
        wind_data = pd.melt(predictions, id_vars=['hour'], value_vars=['wind_speed_10m','wind_gusts_10m'], var_name='Metric', value_name='Value')
        chart = alt.Chart(wind_data).mark_line(point=True).encode(
            x=alt.X('hour:O', title='Hour'),
            y=alt.Y('Value:Q', title='Wind (km/h)'),
            color='Metric:N',
            tooltip=['hour', 'Value', 'Metric']
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

# -------------------- PRECIPITATION CHART --------------------
with tabs[5]:
    st.markdown("<h2>Precipitation Chart</h2>", unsafe_allow_html=True)
    if 'precipitation' in predictions.columns:
        chart = alt.Chart(predictions).mark_bar(color='purple').encode(
            x=alt.X('hour:O', title='Hour'),
            y=alt.Y('precipitation', title='Precipitation (mm)'),
            tooltip=[alt.Tooltip('hour'), alt.Tooltip('precipitation', format='.2f')]
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
