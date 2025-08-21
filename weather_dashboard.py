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
with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Custom header with icon
st.markdown("""
<div class="header">
    <i class="fas fa-cloud-sun header-icon"></i>
    <h1 class="header-title">Hourly Weather Dashboard</h1>
</div>
""", unsafe_allow_html=True)

# -------------------- LOAD HISTORICAL DATA & MODEL --------------------
csv_file = "open-meteo-27.75N85.50E1293mNew.csv"
model_file = "weather_prediction_hourly_model.pkl"

df_hist = pd.DataFrame()  # Initialize an empty DataFrame
model_data = {}  # Initialize an empty dict for model data

# Load historical data with error handling
if not os.path.exists(csv_file):
    st.error(f"Error: Historical data file '{csv_file}' not found. Please ensure it's in the same directory.")
    st.stop()  # Stop the app if the file is not found
try:
    df_hist = pd.read_csv(csv_file)
    df_hist['time'] = pd.to_datetime(df_hist['time'], errors='coerce')
    # Drop rows where 'time' could not be parsed (NaT)
    df_hist.dropna(subset=['time'], inplace=True)
except Exception as e:
    st.error(f"Error loading historical data from '{csv_file}': {e}")
    st.stop()

# Feature engineering (only if df_hist is not empty)
feature_cols = []  # Initialize feature_cols
if not df_hist.empty:
    df_hist['hour'] = df_hist['time'].dt.hour.astype(int)
    df_hist['day_of_week'] = df_hist['time'].dt.dayofweek
    df_hist['month'] = df_hist['time'].dt.month
    df_hist['year'] = df_hist['time'].dt.year
    df_hist['is_weekend'] = df_hist['day_of_week'].isin([5, 6]).astype(int)

    # Define feature columns dynamically based on available numerical columns, excluding targets
    target_columns_potential = [
        'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation',
        'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m'
    ]
    numerical_cols = df_hist.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numerical_cols if col not in target_columns_potential]

    # Fill missing values with median for feature columns
    for col in feature_cols:
        if col in df_hist.columns:  # Ensure column exists before trying to fill
            df_hist[col].fillna(df_hist[col].median(), inplace=True)
else:
    st.info("Historical data is empty or could not be loaded. Some features might be unavailable.")

# Load trained model with error handling
if not os.path.exists(model_file):
    st.error(f"Error: Model file '{model_file}' not found. Please ensure it's in the same directory.")
    st.stop()
try:
    model_data = joblib.load(model_file)
    # Basic check for expected keys in model_data
    if not all(k in model_data for k in ['target_columns', 'models', 'feature_columns']):
        st.error("Error: Model data structure is unexpected. Missing 'target_columns', 'models', or 'feature_columns'.")
        st.stop()
except Exception as e:
    st.error(f"Error loading model from '{model_file}': {e}")
    st.stop()

# -------------------- DATE SELECTION --------------------
min_date = datetime.now().date()
max_date = min_date + timedelta(days=7)

st.sidebar.markdown("""
<div class="sidebar-header">
    <i class="fas fa-sliders-h sidebar-icon"></i>
    <h3 class="sidebar-title">Dashboard Controls</h3>
</div>
""", unsafe_allow_html=True)

selected_date = st.sidebar.date_input(
    "Select Date",
    value=min_date,
    min_value=min_date,
    max_value=max_date
)

# Sidebar for metric selection for hourly cards
display_metric = st.sidebar.selectbox(
    "Select Metric for Hourly Cards",
    ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'precipitation'],
    format_func=lambda x: x.replace('_', ' ').replace('2m', '').replace('10m', '').title()
)

# Mapping for units, display names, and icons
metric_units = {
    'temperature_2m': '°C',
    'relative_humidity_2m': '%',
    'dew_point_2m': '°C',
    'precipitation': 'mm',
    'wind_speed_10m': 'km/h',
    'wind_direction_10m': '°',
    'wind_gusts_10m': 'km/h'
}

metric_display_names = {
    'temperature_2m': 'Temperature',
    'relative_humidity_2m': 'Humidity',
    'dew_point_2m': 'Dew Point',
    'precipitation': 'Precipitation',
    'wind_speed_10m': 'Wind Speed',
    'wind_direction_10m': 'Wind Direction',
    'wind_gusts_10m': 'Wind Gusts'
}

metric_icons = {
    'temperature_2m': 'fas fa-thermometer-half',
    'relative_humidity_2m': 'fas fa-tint',
    'dew_point_2m': 'fas fa-cloud-rain',
    'precipitation': 'fas fa-cloud-showers-heavy',
    'wind_speed_10m': 'fas fa-wind',
    'wind_direction_10m': 'fas fa-compass',
    'wind_gusts_10m': 'fas fa-wind'
}

# -------------------- CREATE FUTURE FEATURES --------------------
def create_future_features_for_date(date, hours=24):
    """
    Creates a DataFrame with future time features for a given date.
    Assumes median values for other feature columns from historical data.
    """
    future_times = [datetime.combine(date, datetime.min.time()) + timedelta(hours=i) for i in range(hours)]
    df_future = pd.DataFrame({'time': future_times})
    df_future['hour'] = df_future['time'].dt.hour.astype(int)
    df_future['day_of_week'] = df_future['time'].dt.dayofweek
    df_future['month'] = df_future['time'].dt.month
    df_future['year'] = df_future['time'].dt.year
    df_future['is_weekend'] = df_future['day_of_week'].isin([5, 6]).astype(int)

    # Populate other feature columns with median from historical data
    # Ensure feature_cols are present in df_hist before using median
    for col in model_data['feature_columns']:
        if col in df_hist.columns and not df_hist[col].empty and col not in df_future.columns:
            df_future[col] = df_hist[col].median()
        elif col not in df_future.columns:
            df_future[col] = 0  # Default to 0 if historical median is not available

    return df_future

# -------------------- PREDICT WEATHER --------------------
def predict_weather(df_future, model_data):
    """
    Predicts weather metrics using the loaded models.
    """
    predictions = pd.DataFrame({'time': df_future['time'], 'hour': df_future['hour']})
    
    # Check if 'feature_columns' exist in df_future before prediction
    if not all(col in df_future.columns for col in model_data['feature_columns']):
        missing_cols = [col for col in model_data['feature_columns'] if col not in df_future.columns]
        st.warning(f"Warning: Missing required feature columns: {missing_cols}. Predictions might be inaccurate.")
        # Add missing columns with default values
        for col in missing_cols:
            df_future[col] = 0

    for target in model_data['target_columns']:
        if target in model_data['models']:
            model = model_data['models'][target]
            predictions[target] = model.predict(df_future[model_data['feature_columns']])
        else:
            st.warning(f"Model for {target} not found in the loaded model data.")
            predictions[target] = 0  # Default value if model not found
    
    return predictions

df_future_features = create_future_features_for_date(selected_date)
predictions = predict_weather(df_future_features, model_data)

# -------------------- CURRENT HOUR WEATHER --------------------
st.markdown("""
<div class="card">
    <div class="card-header">
        <i class="fas fa-clock card-icon"></i>
        <h2 class="card-title">Current Hour Weather</h2>
    </div>
""", unsafe_allow_html=True)

current_hour_dt = datetime.now()
current_hour_value = current_hour_dt.hour

# Filter predictions for the current hour of the selected date
current_weather = predictions[
    (predictions['time'].dt.date == selected_date) &
    (predictions['hour'] == current_hour_value)
]

if not current_weather.empty:
    row = current_weather.iloc[0]
    st.markdown(f"<p><strong>Data for {current_hour_dt.strftime('%Y-%m-%d %H:00')}:</strong></p>", unsafe_allow_html=True)
    
    # Create current weather metrics grid
    st.markdown('<div class="current-weather-grid">', unsafe_allow_html=True)
    
    # Temperature
    st.markdown(f"""
    <div class="weather-metric">
        <i class="{metric_icons['temperature_2m']} metric-icon"></i>
        <span class="metric-value">{row.get('temperature_2m', 0):.1f}</span>
        <span class="metric-unit">{metric_units.get('temperature_2m', '')}</span>
        <span class="metric-label">{metric_display_names['temperature_2m']}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Humidity
    st.markdown(f"""
    <div class="weather-metric">
        <i class="{metric_icons['relative_humidity_2m']} metric-icon"></i>
        <span class="metric-value">{row.get('relative_humidity_2m', 0):.1f}</span>
        <span class="metric-unit">{metric_units.get('relative_humidity_2m', '')}</span>
        <span class="metric-label">{metric_display_names['relative_humidity_2m']}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Dew Point
    st.markdown(f"""
    <div class="weather-metric">
        <i class="{metric_icons['dew_point_2m']} metric-icon"></i>
        <span class="metric-value">{row.get('dew_point_2m', 0):.1f}</span>
        <span class="metric-unit">{metric_units.get('dew_point_2m', '')}</span>
        <span class="metric-label">{metric_display_names['dew_point_2m']}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Precipitation
    st.markdown(f"""
    <div class="weather-metric">
        <i class="{metric_icons['precipitation']} metric-icon"></i>
        <span class="metric-value">{row.get('precipitation', 0):.2f}</span>
        <span class="metric-unit">{metric_units.get('precipitation', '')}</span>
        <span class="metric-label">{metric_display_names['precipitation']}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Wind Speed
    st.markdown(f"""
    <div class="weather-metric">
        <i class="{metric_icons['wind_speed_10m']} metric-icon"></i>
        <span class="metric-value">{row.get('wind_speed_10m', 0):.1f}</span>
        <span class="metric-unit">{metric_units.get('wind_speed_10m', '')}</span>
        <span class="metric-label">{metric_display_names['wind_speed_10m']}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Wind Gusts
    st.markdown(f"""
    <div class="weather-metric">
        <i class="{metric_icons['wind_gusts_10m']} metric-icon"></i>
        <span class="metric-value">{row.get('wind_gusts_10m', 0):.1f}</span>
        <span class="metric-unit">{metric_units.get('wind_gusts_10m', '')}</span>
        <span class="metric-label">{metric_display_names['wind_gusts_10m']}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.write("No prediction available for the current hour on the selected date.")

st.markdown('</div>', unsafe_allow_html=True)  # Close card

# -------------------- HOURLY WEATHER CARDS --------------------
st.markdown(f"""
<div class="card">
    <div class="card-header">
        <i class="fas fa-hourglass-half card-icon"></i>
        <h2 class="card-title">Hourly Weather for {selected_date.strftime('%Y-%m-%d')}</h2>
    </div>
""", unsafe_allow_html=True)

# Create hourly cards
st.markdown('<div class="hourly-cards">', unsafe_allow_html=True)

for idx, row in predictions.iterrows():
    value_to_display = row.get(display_metric, 0)
    delta_text = f"Wind: {row.get('wind_speed_10m', 0):.1f} km/h" if 'wind_speed_10m' in row and display_metric != 'wind_speed_10m' else ""
    
    st.markdown(f"""
    <div class="hourly-card">
        <span class="hourly-time">{int(row['hour']):02d}:00</span>
        <span class="hourly-value">{value_to_display:.1f}</span>
        <span class="hourly-unit">{metric_units.get(display_metric, '')}</span>
        <span class="hourly-delta">{delta_text}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close hourly-cards
st.markdown('</div>', unsafe_allow_html=True)  # Close card

# -------------------- CHARTS --------------------
st.markdown("""
<div class="card">
    <div class="card-header">
        <i class="fas fa-chart-line card-icon"></i>
        <h2 class="card-title">Weather Charts</h2>
    </div>
    <div class="chart-container">
""", unsafe_allow_html=True)

# Temperature chart
if 'temperature_2m' in predictions.columns:
    temp_chart = alt.Chart(predictions).mark_line(point=True).encode(
        x=alt.X('hour:O', title='Hour of Day'),
        y=alt.Y('temperature_2m', title='Temperature (°C)'),
        tooltip=[
            alt.Tooltip('hour', title='Hour'),
            alt.Tooltip('temperature_2m', title='Temp', format='.1f')
        ]
    ).properties(title='Temperature (°C) Prediction', height=300)
    st.altair_chart(temp_chart, use_container_width=True)
else:
    st.warning("Temperature data not available for chart.")

# Humidity chart
if 'relative_humidity_2m' in predictions.columns:
    humidity_chart = alt.Chart(predictions).mark_line(point=True, color='blue').encode(
        x=alt.X('hour:O', title='Hour of Day'),
        y=alt.Y('relative_humidity_2m', title='Relative Humidity (%)'),
        tooltip=[
            alt.Tooltip('hour', title='Hour'),
            alt.Tooltip('relative_humidity_2m', title='Humidity', format='.1f')
        ]
    ).properties(title='Relative Humidity (%) Prediction', height=300)
    st.altair_chart(humidity_chart, use_container_width=True)
else:
    st.warning("Humidity data not available for chart.")

# Wind chart (combining speed and gusts)
if all(col in predictions.columns for col in ['wind_speed_10m', 'wind_gusts_10m']):
    # Prepare data for wind chart
    wind_data = pd.melt(
        predictions, 
        id_vars=['hour'], 
        value_vars=['wind_speed_10m', 'wind_gusts_10m'],
        var_name='Metric', 
        value_name='Value'
    )
    
    wind_chart = alt.Chart(wind_data).mark_line(point=True).encode(
        x=alt.X('hour:O', title='Hour of Day'),
        y=alt.Y('Value:Q', title='Wind Speed (km/h)'),
        color=alt.Color('Metric:N', legend=alt.Legend(title="Wind Metric")),
        tooltip=[
            alt.Tooltip('hour', title='Hour'),
            alt.Tooltip('Value:Q', title='Value', format='.1f'),
            alt.Tooltip('Metric:N', title='Metric')
        ]
    ).properties(title='Wind Speed & Gusts (km/h) Prediction', height=300)
    st.altair_chart(wind_chart, use_container_width=True)
else:
    st.warning("Wind data not available for chart.")

# Precipitation chart
if 'precipitation' in predictions.columns:
    precip_chart = alt.Chart(predictions).mark_bar(color='purple').encode(
        x=alt.X('hour:O', title='Hour of Day'),
        y=alt.Y('precipitation', title='Precipitation (mm)'),
        tooltip=[
            alt.Tooltip('hour', title='Hour'),
            alt.Tooltip('precipitation', title='Precip', format='.2f')
        ]
    ).properties(title='Precipitation (mm) Prediction', height=300)
    st.altair_chart(precip_chart, use_container_width=True)
else:
    st.warning("Precipitation data not available for chart.")

st.markdown('</div></div>', unsafe_allow_html=True)  # Close chart-container and card