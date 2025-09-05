# weather_dashboard.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
import altair as alt
import requests_cache
from retry_requests import retry
import openmeteo_requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import holidays

st.set_page_config(page_title="Kathmandu Weather Dashboard", layout="wide")
st.title("🌤️ Kathmandu Weather Dashboard — Historical Analysis & Accurate Forecasts")

# -------------------- LOAD HISTORICAL DATA --------------------
@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv("open-meteo-27.75N85.50E1293mNew.csv")
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        
        # Add time-based features
        df['hour'] = df['time'].dt.hour
        df['day'] = df['time'].dt.day
        df['month'] = df['time'].dt.month
        df['weekday'] = df['time'].dt.weekday
        df['day_of_year'] = df['time'].dt.dayofyear
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
        
        # Add holiday information for Nepal
        np_holidays = holidays.Nepal()
        df['is_holiday'] = df['time'].apply(lambda x: x in np_holidays).astype(int)
        
        return df
    except Exception as e:
        st.error(f"Error loading historical data: {e}")
        return None

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load("hourly_weather_model.pkl")
        all_prophet_models = joblib.load("prophet_kathmandu_all_models.pkl")
        return rf_model, all_prophet_models
    except:
        st.error("Model files not found. Please run train_weather_model.py first.")
        return None, None

# Load data and models
historical_data = load_historical_data()
rf_model, all_prophet_models = load_models()

# -------------------- FETCH LIVE WEATHER --------------------
@st.cache_data(ttl=600)
def fetch_live_weather():
    # Setup cache & retry
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 27.7017,
        "longitude": 85.3206,
        "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
                   "precipitation", "visibility", "surface_pressure",
                   "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
        "models": "best_match",
        "current": ["temperature_2m", "relative_humidity_2m", "apparent_temperature",
                    "precipitation", "rain", "showers", "weather_code",
                    "cloud_cover", "wind_speed_10m", "wind_direction_10m",
                    "wind_gusts_10m"],
        "past_days": 2,
        "forecast_days": 7
    }

    responses = client.weather_api(url, params=params)
    response = responses[0]

    # Current weather
    current = response.Current()
    current_weather = {
        "temperature_2m": current.Variables(0).Value(),
        "relative_humidity_2m": current.Variables(1).Value(),
        "apparent_temperature": current.Variables(2).Value(),
        "precipitation": current.Variables(3).Value(),
        "rain": current.Variables(4).Value(),
        "showers": current.Variables(5).Value(),
        "weather_code": current.Variables(6).Value(),
        "cloud_cover": current.Variables(7).Value(),
        "wind_speed_10m": current.Variables(8).Value(),
        "wind_direction_10m": current.Variables(9).Value(),
        "wind_gusts_10m": current.Variables(10).Value(),
        "time": current.Time()
    }

    # Hourly forecast
    hourly = response.Hourly()
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }
    for idx, var in enumerate(["temperature_2m", "relative_humidity_2m", "dew_point_2m",
                               "precipitation", "visibility", "surface_pressure",
                               "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"]):
        hourly_data[var] = hourly.Variables(idx).ValuesAsNumpy()

    hourly_df = pd.DataFrame(hourly_data)
    return current_weather, hourly_df

current_weather, hourly_df = fetch_live_weather()

# -------------------- SIDEBAR CONTROLS --------------------
st.sidebar.header("Dashboard Controls")
view_option = st.sidebar.radio("Select View", ["Current Weather", "Historical Analysis", "Forecasts"])

# Date range selector for historical data
if historical_data is not None:
    min_date = historical_data['time'].min().date()
    max_date = historical_data['time'].max().date()
    default_end = min(max_date, datetime.now().date())
    default_start = max(min_date, default_end - timedelta(days=30))
    
    historical_start = st.sidebar.date_input("Start Date", value=default_start, min_value=min_date, max_value=max_date)
    historical_end = st.sidebar.date_input("End Date", value=default_end, min_value=min_date, max_value=max_date)
    
    if historical_start > historical_end:
        st.sidebar.error("Error: End date must be after start date.")
        historical_start, historical_end = default_start, default_end

# -------------------- CURRENT WEATHER VIEW --------------------
if view_option == "Current Weather":
    st.subheader("🌡️ Current Weather Conditions")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Temperature", f"{current_weather['temperature_2m']:.1f} °C")
    col2.metric("Feels Like", f"{current_weather['apparent_temperature']:.1f} °C")
    col3.metric("Humidity", f"{current_weather['relative_humidity_2m']:.1f}%")
    col4.metric("Wind Speed", f"{current_weather['wind_speed_10m']:.1f} km/h")
    
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Precipitation", f"{current_weather['precipitation']:.1f} mm")
    col6.metric("Cloud Cover", f"{current_weather['cloud_cover']:.1f}%")
    col7.metric("Wind Direction", f"{current_weather['wind_direction_10m']:.0f}°")
    col8.metric("Wind Gusts", f"{current_weather['wind_gusts_10m']:.1f} km/h")
    
    # Weather condition based on weather code
    weather_codes = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
        55: "Dense drizzle", 56: "Light freezing drizzle", 57: "Dense freezing drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        66: "Light freezing rain", 67: "Heavy freezing rain", 71: "Slight snow fall",
        73: "Moderate snow fall", 75: "Heavy snow fall", 77: "Snow grains",
        80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        85: "Slight snow showers", 86: "Heavy snow showers", 95: "Thunderstorm",
        96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
    }
    
    weather_desc = weather_codes.get(int(current_weather['weather_code']), "Unknown")
    st.info(f"**Current conditions:** {weather_desc}")
    
    # Hourly forecast chart
    st.subheader("📈 Hourly Forecast (Next 24 Hours)")
    hourly_next24 = hourly_df[['date', 'temperature_2m', 'relative_humidity_2m', 
                              'precipitation', 'wind_speed_10m']].head(24).copy()
    hourly_next24['hour'] = hourly_next24['date'].dt.strftime("%H:%M")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Temperature", "Humidity", "Precipitation", "Wind"])
    
    with tab1:
        chart = alt.Chart(hourly_next24).mark_line(point=True).encode(
            x=alt.X('hour:N', title='Hour'),
            y=alt.Y('temperature_2m:Q', title='Temperature (°C)'),
            tooltip=['hour', 'temperature_2m']
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
    
    with tab2:
        chart = alt.Chart(hourly_next24).mark_line(point=True, color='green').encode(
            x=alt.X('hour:N', title='Hour'),
            y=alt.Y('relative_humidity_2m:Q', title='Humidity (%)'),
            tooltip=['hour', 'relative_humidity_2m']
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
    
    with tab3:
        chart = alt.Chart(hourly_next24).mark_bar(color='red').encode(
            x=alt.X('hour:N', title='Hour'),
            y=alt.Y('precipitation:Q', title='Precipitation (mm)'),
            tooltip=['hour', 'precipitation']
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
    
    with tab4:
        chart = alt.Chart(hourly_next24).mark_line(point=True, color='purple').encode(
            x=alt.X('hour:N', title='Hour'),
            y=alt.Y('wind_speed_10m:Q', title='Wind Speed (km/h)'),
            tooltip=['hour', 'wind_speed_10m']
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

# -------------------- HISTORICAL ANALYSIS VIEW --------------------
elif view_option == "Historical Analysis" and historical_data is not None:
    st.subheader("📊 Historical Weather Analysis")
    
    # Filter historical data based on selected date range
    filtered_data = historical_data[
        (historical_data['time'].dt.date >= historical_start) & 
        (historical_data['time'].dt.date <= historical_end)
    ]
    
    if filtered_data.empty:
        st.warning("No data available for the selected date range.")
    else:
        # Select variable to analyze
        analysis_var = st.selectbox(
            "Select variable to analyze:",
            ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]
        )
        
        # Display summary statistics
        st.write("**Summary Statistics**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average", f"{filtered_data[analysis_var].mean():.2f}")
        col2.metric("Maximum", f"{filtered_data[analysis_var].max():.2f}")
        col3.metric("Minimum", f"{filtered_data[analysis_var].min():.2f}")
        col4.metric("Std Deviation", f"{filtered_data[analysis_var].std():.2f}")
        
        # Time series chart
        st.write("**Time Series**")
        chart = alt.Chart(filtered_data).mark_line().encode(
            x=alt.X('time:T', title='Date'),
            y=alt.Y(f'{analysis_var}:Q', title=analysis_var.replace('_', ' ').title()),
            tooltip=['time', analysis_var]
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)
        
        # Daily averages
        st.write("**Daily Averages**")
        daily_avg = filtered_data.resample('D', on='time').mean().reset_index()
        chart = alt.Chart(daily_avg).mark_bar().encode(
            x=alt.X('time:T', title='Date'),
            y=alt.Y(f'{analysis_var}:Q', title=analysis_var.replace('_', ' ').title()),
            tooltip=['time', analysis_var]
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
        
        # Hourly averages
        st.write("**Hourly Averages**")
        hourly_avg = filtered_data.groupby('hour').mean().reset_index()
        chart = alt.Chart(hourly_avg).mark_line(point=True).encode(
            x=alt.X('hour:Q', title='Hour of Day'),
            y=alt.Y(f'{analysis_var}:Q', title=analysis_var.replace('_', ' ').title()),
            tooltip=['hour', analysis_var]
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
        
        # Monthly averages
        st.write("**Monthly Averages**")
        monthly_avg = filtered_data.groupby('month').mean().reset_index()
        chart = alt.Chart(monthly_avg).mark_bar().encode(
            x=alt.X('month:O', title='Month'),
            y=alt.Y(f'{analysis_var}:Q', title=analysis_var.replace('_', ' ').title()),
            tooltip=['month', analysis_var]
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

# -------------------- FORECASTS VIEW --------------------
elif view_option == "Forecasts":
    st.subheader("🔮 Weather Forecasts")
    
    if all_prophet_models is None:
        st.error("Prophet models not available. Please run train_weather_model.py first.")
    else:
        # Daily forecast with Prophet
        feature_choice = st.selectbox(
            "Select variable to forecast:",
            ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]
        )
        
        daily_model = all_prophet_models[feature_choice]
        future = daily_model.make_future_dataframe(periods=7, freq="D")
        forecast = daily_model.predict(future)
        
        daily_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)
        daily_forecast['ds'] = pd.to_datetime(daily_forecast['ds']).dt.date
        
        st.write("**7-Day Forecast**")
        st.table(daily_forecast.rename(columns={
            'ds': 'Date', 
            'yhat': f'{feature_choice} Forecast',
            'yhat_lower': 'Lower Bound',
            'yhat_upper': 'Upper Bound'
        }))
        
        # Forecast chart with confidence intervals
        chart = alt.Chart(daily_forecast).mark_line(point=True).encode(
            x=alt.X('ds:T', title='Date'),
            y=alt.Y('yhat:Q', title=f'{feature_choice} Forecast'),
            tooltip=['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        ).properties(height=300)
        
        # Add confidence interval
        band = alt.Chart(daily_forecast).mark_area(opacity=0.3).encode(
            x='ds:T',
            y='yhat_lower:Q',
            y2='yhat_upper:Q'
        )
        
        st.altair_chart(chart + band, use_container_width=True)
        
        # Compare with historical data if available
        if historical_data is not None:
            st.write("**Comparison with Historical Averages**")
            
            # Get historical averages for the same dates
            historical_compare = historical_data.copy()
            historical_compare['day_month'] = historical_compare['time'].dt.strftime('%m-%d')
            historical_avg = historical_compare.groupby('day_month')[feature_choice].mean().reset_index()
            
            # Create comparison data
            forecast_compare = daily_forecast.copy()
            forecast_compare['day_month'] = forecast_compare['ds'].apply(lambda x: x.strftime('%m-%d'))
            
            comparison_df = forecast_compare.merge(historical_avg, on='day_month', how='left')
            comparison_df = comparison_df.rename(columns={feature_choice: 'historical_avg'})
            
            if not comparison_df.empty:
                # Melt for Altair
                comparison_melt = comparison_df.melt(id_vars=['ds'], value_vars=['yhat', 'historical_avg'], 
                                                    var_name='Type', value_name='Value')
                comparison_melt['Type'] = comparison_melt['Type'].replace({
                    'yhat': 'Forecast', 
                    'historical_avg': 'Historical Average'
                })
                
                comp_chart = alt.Chart(comparison_melt).mark_line(point=True).encode(
                    x='ds:T',
                    y='Value:Q',
                    color='Type:N',
                    tooltip=['ds', 'Type', 'Value']
                ).properties(height=300)
                
                st.altair_chart(comp_chart, use_container_width=True)
                
                # Calculate difference
                comparison_df['difference'] = comparison_df['yhat'] - comparison_df['historical_avg']
                st.write("**Difference from Historical Average**")
                st.table(comparison_df[['ds', 'yhat', 'historical_avg', 'difference']].rename(columns={
                    'ds': 'Date', 
                    'yhat': 'Forecast',
                    'historical_avg': 'Historical Avg',
                    'difference': 'Difference'
                }))

# -------------------- FOOTER --------------------
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Data Sources:**
    - Historical data: open-meteo-27.75N85.50E1293mNew.csv
    - Current weather: Open-Meteo API
    - Forecasts: Prophet models trained on historical data
    """
)

# Show data info in sidebar
if historical_data is not None:
    st.sidebar.write(f"Historical data range: {historical_data['time'].min().date()} to {historical_data['time'].max().date()}")
    st.sidebar.write(f"Total records: {len(historical_data):,}")