# weather_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import altair as alt
import os
import math

# -------------------- PAGE SETUP --------------------
st.set_page_config(
    page_title="Weather Forecast Dashboard",
    layout="wide",
    page_icon="🌤️",
    initial_sidebar_state="expanded"
)

# Load CSS
if os.path.exists('styles.css'):
    with open('styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Custom header
st.markdown("""
<div class="header">
    <i class="fas fa-cloud-sun header-icon"></i>
    <h1 class="header-title">Weather Forecast Dashboard</h1>
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
    
    # Filter to only include data from the last 30 days for historical charts
    thirty_days_ago = datetime.now() - timedelta(days=30)
    df_hist_recent = df_hist[df_hist['time'] >= thirty_days_ago].copy()
    
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# Feature engineering for prediction
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

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <i class="fas fa-cog sidebar-icon"></i>
        <span class="sidebar-title">Dashboard Settings</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Date selection
    min_date = datetime.now().date()
    max_date = min_date + timedelta(days=7)
    selected_date = st.date_input("Select Forecast Date", value=min_date, min_value=min_date, max_value=max_date)
    
    # Historical data period selection
    hist_days = st.slider("Historical Data Period (days)", min_value=7, max_value=90, value=30)
    
    # Location info
    st.markdown("---")
    st.markdown("""
    <div class="sidebar-section">
        <h3><i class="fas fa-map-marker-alt"></i> Location</h3>
        <p>Kathmandu, Nepal<br>27.75°N, 85.50°E<br>Elevation: 1293m</p>
    </div>
    """, unsafe_allow_html=True)

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

# -------------------- PREDICTION FUNCTIONS --------------------
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

# Generate predictions
df_future_features = create_future_features_for_date(selected_date)
predictions = predict_weather(df_future_features, model_data)

# Filter historical data based on selected period
hist_start_date = datetime.now() - timedelta(days=hist_days)
df_hist_filtered = df_hist[df_hist['time'] >= hist_start_date].copy()

# -------------------- WIND DIRECTION ROSE FUNCTION --------------------
def create_wind_rose_data(df):
    if 'wind_direction_10m' not in df.columns or 'wind_speed_10m' not in df.columns:
        return pd.DataFrame()
    
    # Create direction bins (16 compass points)
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    direction_bins = np.linspace(0, 360, 17)  # 16 directions + 1 for wrap-around
    
    # Create speed bins
    speed_bins = [0, 5, 10, 15, 20, 25, 30, 100]  # km/h
    speed_labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30+']
    
    # Filter out rows with missing wind data
    wind_data = df[['wind_direction_10m', 'wind_speed_10m']].dropna()
    
    if wind_data.empty:
        return pd.DataFrame()
    
    # Assign direction categories
    wind_data['direction_cat'] = pd.cut(
        wind_data['wind_direction_10m'], 
        bins=direction_bins, 
        labels=directions, 
        include_lowest=True
    )
    
    # Assign speed categories
    wind_data['speed_cat'] = pd.cut(
        wind_data['wind_speed_10m'], 
        bins=speed_bins, 
        labels=speed_labels, 
        include_lowest=True
    )
    
    # Count frequency
    wind_rose_data = wind_data.groupby(['direction_cat', 'speed_cat']).size().reset_index(name='count')
    wind_rose_data['percentage'] = (wind_rose_data['count'] / wind_rose_data['count'].sum()) * 100
    
    return wind_rose_data

# Create wind rose data
wind_rose_data = create_wind_rose_data(df_hist_filtered)

# -------------------- PREDICTION SUMMARY --------------------
st.markdown("## Weather Forecast Summary")
current_hour_dt = datetime.now()
current_hour_value = current_hour_dt.hour
current_weather = predictions[
    (predictions['time'].dt.date == selected_date) &
    (predictions['hour'] == current_hour_value)
]

if not current_weather.empty:
    row = current_weather.iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-icon temp-icon">
                <i class="{metric_icons['temperature_2m']}"></i>
            </div>
            <div class="summary-content">
                <div class="summary-value">{row.get('temperature_2m', 0):.1f}{metric_units['temperature_2m']}</div>
                <div class="summary-label">Temperature</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-icon humidity-icon">
                <i class="{metric_icons['relative_humidity_2m']}"></i>
            </div>
            <div class="summary-content">
                <div class="summary-value">{row.get('relative_humidity_2m', 0):.1f}{metric_units['relative_humidity_2m']}</div>
                <div class="summary-label">Humidity</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-icon wind-icon">
                <i class="{metric_icons['wind_speed_10m']}"></i>
            </div>
            <div class="summary-content">
                <div class="summary-value">{row.get('wind_speed_10m', 0):.1f}{metric_units['wind_speed_10m']}</div>
                <div class="summary-label">Wind Speed</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-icon precip-icon">
                <i class="{metric_icons['precipitation']}"></i>
            </div>
            <div class="summary-content">
                <div class="summary-value">{row.get('precipitation', 0):.2f}{metric_units['precipitation']}</div>
                <div class="summary-label">Precipitation</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# -------------------- TABS --------------------
tabs = st.tabs([
    "Forecast", "Historical Wind Analysis"
])

# -------------------- FORECAST TAB --------------------
with tabs[0]:
    st.markdown(f"<h2>Weather Forecast for {selected_date.strftime('%B %d, %Y')}</h2>", unsafe_allow_html=True)
    
    # Create forecast charts in columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Temperature Forecast
        if 'temperature_2m' in predictions.columns:
            st.markdown("#### Temperature Forecast")
            chart = alt.Chart(predictions).mark_line(
                point=True, 
                color='#FF4B4B',
                strokeWidth=3
            ).encode(
                x=alt.X('hoursminutes(time):T', title='Time', axis=alt.Axis(format='%H:%M')),
                y=alt.Y('temperature_2m:Q', title='Temperature (°C)', scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip('hoursminutes(time):T', title='Time'), 
                         alt.Tooltip('temperature_2m:Q', title='Temperature', format='.1f')]
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
    
    with col2:
        # Precipitation Forecast
        if 'precipitation' in predictions.columns:
            st.markdown("#### Precipitation Forecast")
            chart = alt.Chart(predictions).mark_bar(
                color='#6A4C93',
                cornerRadiusTopLeft=3,
                cornerRadiusTopRight=3
            ).encode(
                x=alt.X('hoursminutes(time):T', title='Time', axis=alt.Axis(format='%H:%M')),
                y=alt.Y('precipitation:Q', title='Precipitation (mm)'),
                tooltip=[alt.Tooltip('hoursminutes(time):T', title='Time'), 
                         alt.Tooltip('precipitation:Q', title='Precipitation', format='.2f')]
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
    
    # Hourly Forecast Details
    st.markdown("#### Hourly Forecast Details")
    st.markdown('<div class="hourly-cards-grid">', unsafe_allow_html=True)
    
    for idx, row in predictions.iterrows():
        temp = row.get('temperature_2m', 0)
        precip = row.get('precipitation', 0)
        
        if precip > 0.5:
            weather_icon = "fas fa-cloud-showers-heavy"
            weather_desc = "Rainy"
        elif temp > 25:
            weather_icon = "fas fa-sun"
            weather_desc = "Sunny"
        elif temp > 15:
            weather_icon = "fas fa-cloud-sun"
            weather_desc = "Partly Cloudy"
        else:
            weather_icon = "fas fa-cloud"
            weather_desc = "Cloudy"
            
        st.markdown(f"""
        <div class="hourly-card">
            <div class="hourly-time">{int(row['hour']):02d}:00</div>
            <div class="hourly-icon"><i class="{weather_icon}"></i></div>
            <div class="hourly-desc">{weather_desc}</div>
            <div class="hourly-temp">{row.get('temperature_2m',0):.1f}°</div>
            <div class="hourly-details">
                <div class="detail-item"><i class="fas fa-tint"></i> {row.get('relative_humidity_2m',0):.0f}%</div>
                <div class="detail-item"><i class="fas fa-wind"></i> {row.get('wind_speed_10m',0):.0f} km/h</div>
                <div class="detail-item"><i class="fas fa-cloud-rain"></i> {row.get('precipitation',0):.1f} mm</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- HISTORICAL WIND ANALYSIS TAB --------------------
with tabs[1]:
    st.markdown(f"<h2>Historical Wind Analysis (Last {hist_days} Days)</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Wind Speed Time Series
        if 'wind_speed_10m' in df_hist_filtered.columns:
            st.markdown("#### Wind Speed Over Time")
            chart = alt.Chart(df_hist_filtered).mark_line(
                color='#2E86AB',
                strokeWidth=2
            ).encode(
                x=alt.X('time:T', title='Date'),
                y=alt.Y('wind_speed_10m:Q', title='Wind Speed (km/h)'),
                tooltip=[alt.Tooltip('time:T', title='Time'), 
                         alt.Tooltip('wind_speed_10m:Q', title='Wind Speed', format='.1f')]
            ).properties(height=400)
            st.altair_chart(chart, use_container_width=True)
    
    with col2:
        # Wind Direction Rose
        if not wind_rose_data.empty:
            st.markdown("#### Wind Direction Rose")
            
            # Create polar coordinates for wind rose
            directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                         'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
            direction_angles = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5,
                              180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5]
            
            # Create base data for all directions
            base_data = pd.DataFrame({
                'direction_cat': directions * len(wind_rose_data['speed_cat'].unique()),
                'angle': direction_angles * len(wind_rose_data['speed_cat'].unique()),
                'speed_cat': sorted(wind_rose_data['speed_cat'].unique().tolist() * 16)
            })
            
            # Merge with actual data
            wind_rose_full = base_data.merge(wind_rose_data, on=['direction_cat', 'speed_cat'], how='left')
            wind_rose_full['percentage'] = wind_rose_full['percentage'].fillna(0)
            
            # Create wind rose chart
            chart = alt.Chart(wind_rose_full).mark_arc(innerRadius=20, stroke="#fff").encode(
                theta=alt.Theta("angle:Q", stack=True),
                radius=alt.Radius("percentage:Q", scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)),
                color=alt.Color("speed_cat:N", legend=alt.Legend(title="Wind Speed (km/h)")),
                tooltip=['direction_cat', 'speed_cat', 'percentage']
            ).properties(
                width=400,
                height=400
            )
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No wind direction data available for the selected period.")
    
    # Wind Statistics
    st.markdown("#### Wind Statistics")
    if 'wind_speed_10m' in df_hist_filtered.columns and 'wind_direction_10m' in df_hist_filtered.columns:
        wind_stats = df_hist_filtered[['wind_speed_10m', 'wind_direction_10m']].describe()
        st.dataframe(wind_stats.style.format("{:.2f}"))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Wind Speed", f"{df_hist_filtered['wind_speed_10m'].mean():.1f} km/h")
        with col2:
            st.metric("Max Wind Speed", f"{df_hist_filtered['wind_speed_10m'].max():.1f} km/h")
        with col3:
            # Calculate prevailing wind direction
            if not wind_rose_data.empty:
                prevailing_dir = wind_rose_data.groupby('direction_cat')['count'].sum().idxmax()
                st.metric("Prevailing Direction", prevailing_dir)

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Weather forecast powered by machine learning • Historical data from Open-Meteo • Updated: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)