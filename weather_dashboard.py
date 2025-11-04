#weather_dashboard
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
import altair as alt
import holidays

st.set_page_config(page_title="Kathmandu Weather Dashboard", layout="wide")
st.title("Kathmandu Weather Dashboard")

# -----------------------------
# LOAD HISTORICAL DATA
# -----------------------------
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

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    try:
        # Load Random Forest models
        rf_models = joblib.load("hourly_weather_model.pkl")
        
        # Get feature names from the first model
        first_model = list(rf_models.values())[0]
        model_features = list(first_model.feature_names_in_)
        
        # Load Prophet models
        prophet_models = joblib.load("prophet_models.pkl")
        
        return rf_models, None, model_features, prophet_models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run train_weather_model.py first to train the models.")
        return None, None, None, None

# Load data and models
historical_data = load_historical_data()
rf_models, scaler, model_features, prophet_models = load_models()

# Weather icon mapping using HTML/CSS icons
def get_weather_icon(weather_type):
    """Return HTML for weather icons"""
    icon_map = {
        'Clear': '☀️',
        'Sunny': '☀️',
        'Mainly Clear': '🌤️',
        'Partly Cloudy': '⛅',
        'Cloudy': '☁️',
        'Overcast': '☁️',
        'Foggy': '🌫️',
        'Mist': '🌫️',
        'Light Rain': '🌦️',
        'Rain': '🌧️',
        'Heavy Rain': '⛈️',
        'Light Snow': '🌨️',
        'Snow': '❄️',
        'Heavy Snow': '☃️',
        'Thunderstorm': '⛈️',
        'Drizzle': '💧',
        'Showers': '🌦️',
        'Windy': '💨',
        'Breezy': '💨',
        'Humid': '💦',
        'Hot': '🔥',
        'Warm': '😊',
        'Mild': '😐',
        'Cool': '🍃',
        'Cold': '🥶',
        'Freezing': '🧊'
    }
    return icon_map.get(weather_type, '🌈')

# Weather color mapping for better visual distinction
def get_weather_color(weather_type):
    """Return color for weather type"""
    color_map = {
        'Clear': '#FFD700',  # Gold
        'Sunny': '#FFA500',  # Orange
        'Mainly Clear': '#87CEEB',  # Sky Blue
        'Partly Cloudy': '#B0C4DE',  # Light Steel Blue
        'Cloudy': '#696969',  # Dim Gray
        'Overcast': '#808080',  # Gray
        'Foggy': '#D3D3D3',  # Light Gray
        'Mist': '#E6E6FA',  # Lavender
        'Light Rain': '#6495ED',  # Cornflower Blue
        'Rain': '#0000FF',  # Blue
        'Heavy Rain': '#000080',  # Navy
        'Light Snow': '#F0F8FF',  # Alice Blue
        'Snow': '#FFFFFF',  # White
        'Heavy Snow': '#E6E6FA',  # Lavender
        'Thunderstorm': '#4B0082',  # Indigo
        'Drizzle': '#ADD8E6',  # Light Blue
        'Showers': '#1E90FF',  # Dodger Blue
        'Windy': '#A9A9A9',  # Dark Gray
        'Breezy': '#C0C0C0',  # Silver
        'Humid': '#20B2AA',  # Light Sea Green
        'Hot': '#FF4500',  # Orange Red
        'Warm': '#FF8C00',  # Dark Orange
        'Mild': '#32CD32',  # Lime Green
        'Cool': '#00CED1',  # Dark Turquoise
        'Cold': '#00BFFF',  # Deep Sky Blue
        'Freezing': '#0000CD'  # Medium Blue
    }
    return color_map.get(weather_type, '#808080')

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("Dashboard Controls")
view_option = st.sidebar.radio("Select View", ["Current Weather", "Historical Analysis", "Forecasts", "Hourly Predictions"])

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

# -----------------------------
# CURRENT WEATHER VIEW
# -----------------------------
if view_option == "Current Weather":
    st.subheader("Current Weather Conditions")
    if historical_data is not None:
        current = historical_data.iloc[-1]
        
        # Display current weather metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Temperature", f"{current['temperature_2m']:.1f} °C")
        col2.metric("Feels Like", f"{current['temperature_2m']:.1f} °C")
        col3.metric("Humidity", f"{current['relative_humidity_2m']:.1f}%")
        col4.metric("Wind Speed", f"{current['wind_speed_10m']:.1f} km/h")
        
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Precipitation", f"{current['precipitation']:.1f} mm")
        col6.metric("Cloud Cover", f"{current.get('cloud_cover', 0):.1f}%")
        col7.metric("Wind Direction", f"{current['wind_direction_10m']:.0f}°")
        col8.metric("Wind Gusts", f"{current['wind_gusts_10m']:.1f} km/h")
        
        # Weather description with icon
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
        current_weather_code = current.get('weather_code', 0)
        current_weather_desc = weather_codes.get(current_weather_code, 'Unknown')
        icon = get_weather_icon(current_weather_desc.split()[0])
        
        st.info(f"**Current conditions:** {icon} {current_weather_desc}")

# -----------------------------
# HISTORICAL ANALYSIS VIEW
# -----------------------------
elif view_option == "Historical Analysis" and historical_data is not None:
    st.subheader("Historical Weather Analysis")
    
    filtered_data = historical_data[
        (historical_data['time'].dt.date >= historical_start) &
        (historical_data['time'].dt.date <= historical_end)
    ]
    
    if filtered_data.empty:
        st.warning("No data available for the selected date range.")
    else:
        analysis_var = st.selectbox(
            "Select variable to analyze:",
            ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]
        )
        
        st.write("**Summary Statistics**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average", f"{filtered_data[analysis_var].mean():.2f}")
        col2.metric("Maximum", f"{filtered_data[analysis_var].max():.2f}")
        col3.metric("Minimum", f"{filtered_data[analysis_var].min():.2f}")
        col4.metric("Std Deviation", f"{filtered_data[analysis_var].std():.2f}")
        
        st.write("**Time Series**")
        chart = alt.Chart(filtered_data).mark_line().encode(
            x=alt.X('time:T', title='Date'),
            y=alt.Y(f'{analysis_var}:Q', title=analysis_var.replace('_',' ').title()),
            tooltip=['time', analysis_var]
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)

# -----------------------------
# HOURLY PREDICTIONS VIEW
# -----------------------------
elif view_option == "Hourly Predictions":
    st.subheader("Hourly Predictions (Next 24 Hours)")
    if rf_models is not None and model_features is not None and historical_data is not None:
        try:
            hourly_preds = []
            # Create a working copy of the last row
            last = historical_data.iloc[-1].copy()
            np_holidays = holidays.Nepal()
            
            for i in range(24):
                # Prepare input features
                X_input_dict = {}
                
                # Add basic time features
                current_time = last['time'] + pd.Timedelta(hours=1) if i > 0 else last['time']
                X_input_dict['hour'] = current_time.hour
                X_input_dict['day'] = current_time.day
                X_input_dict['month'] = current_time.month
                X_input_dict['weekday'] = current_time.weekday()
                X_input_dict['day_of_year'] = current_time.timetuple().tm_yday
                X_input_dict['is_weekend'] = int(current_time.weekday() in [5, 6])
                X_input_dict['is_holiday'] = int(current_time in np_holidays)
                
                # Add lag features using historical data
                for feature in model_features:
                    if '_lag_' in feature:
                        base_col = '_'.join(feature.split('_')[:-2])
                        lag_hours = int(feature.split('_')[-1])
                        
                        # Calculate the timestamp for the lag
                        lag_time = current_time - pd.Timedelta(hours=lag_hours)
                        
                        # Find the historical value for that timestamp
                        historical_match = historical_data[historical_data['time'] == lag_time]
                        if not historical_match.empty:
                            X_input_dict[feature] = historical_match[base_col].iloc[0]
                        else:
                            # If no exact match, use the most recent available value
                            earlier_data = historical_data[historical_data['time'] <= lag_time]
                            if not earlier_data.empty:
                                X_input_dict[feature] = earlier_data[base_col].iloc[-1]
                            else:
                                X_input_dict[feature] = historical_data[base_col].iloc[0]
                    
                    elif 'rolling' in feature:
                        # For rolling features, use historical averages
                        parts = feature.split('_')
                        base_col = '_'.join(parts[:2])  # e.g., 'temp_rolling_mean_3' -> 'temp_rolling'
                        window = int(parts[-1])
                        
                        # Use recent historical data for rolling calculations
                        recent_data = historical_data.tail(window * 2)  # Get enough data for calculation
                        if not recent_data.empty and base_col.split('_')[0] in recent_data.columns:
                            if 'mean' in feature:
                                X_input_dict[feature] = recent_data[base_col.split('_')[0]].mean()
                            elif 'std' in feature:
                                X_input_dict[feature] = recent_data[base_col.split('_')[0]].std()
                        else:
                            X_input_dict[feature] = historical_data[base_col.split('_')[0]].mean() if base_col.split('_')[0] in historical_data.columns else 0
                
                # Create input DataFrame with proper feature order
                X_input = pd.DataFrame([X_input_dict])[model_features]
                
                # Make predictions
                pred_row = {'time': current_time + pd.Timedelta(hours=1)}
                
                # Predict numerical values
                for target in ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'precipitation']:
                    if target in rf_models:
                        pred = rf_models[target].predict(X_input)[0]
                        pred_row[target] = pred
                
                # Predict weather type
                if 'weather_type' in rf_models:
                    weather_pred = rf_models['weather_type'].predict(X_input)[0]
                    pred_row['weather_type'] = weather_pred
                    pred_row['weather_icon'] = get_weather_icon(weather_pred)
                    pred_row['weather_color'] = get_weather_color(weather_pred)
                
                hourly_preds.append(pred_row)
                
                # Update the last values for next iteration (only for base features)
                for target in ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'precipitation']:
                    if target in last:
                        last[target] = pred_row[target]

            # Display results
            hourly_df = pd.DataFrame(hourly_preds)
            st.write("**Next 24 Hours Forecast**")
            
            # Create a styled dataframe with weather icons
            display_data = []
            for _, row in hourly_df.iterrows():
                display_row = {
                    'Time': row['time'].strftime('%Y-%m-%d %H:%M'),
                    'Temperature': f"{row['temperature_2m']:.1f}°C",
                    'Humidity': f"{row['relative_humidity_2m']:.1f}%",
                    'Wind Speed': f"{row['wind_speed_10m']:.1f} km/h",
                    'Precipitation': f"{row['precipitation']:.2f} mm",
                    'Weather': f"{row.get('weather_icon', '')} {row.get('weather_type', 'Unknown')}"
                }
                display_data.append(display_row)
            
            display_df = pd.DataFrame(display_data)
            st.dataframe(display_df, use_container_width=True)
            
            # Add charts and visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Temperature Forecast**")
                temp_chart = alt.Chart(hourly_df).mark_line(color='red').encode(
                    x=alt.X('time:T', title='Time'),
                    y=alt.Y('temperature_2m:Q', title='Temperature (°C)'),
                    tooltip=['time', 'temperature_2m']
                ).properties(height=300)
                st.altair_chart(temp_chart, use_container_width=True)
            
            with col2:
                st.write("**Weather Conditions Timeline**")
                if 'weather_type' in hourly_df.columns and 'weather_color' in hourly_df.columns:
                    # Create a timeline chart with colored points for weather types
                    timeline_chart = alt.Chart(hourly_df).mark_circle(size=100).encode(
                        x=alt.X('time:T', title='Time'),
                        y=alt.Y('temperature_2m:Q', title='Temperature (°C)'),
                        color=alt.Color('weather_type:N', scale=alt.Scale(
                            domain=list(set(hourly_df['weather_type'])),
                            range=[get_weather_color(wt) for wt in set(hourly_df['weather_type'])]
                        )),
                        tooltip=['time', 'temperature_2m', 'weather_type']
                    ).properties(height=300)
                    st.altair_chart(timeline_chart, use_container_width=True)
            
            # Weather distribution
            st.write("**Weather Distribution**")
            if 'weather_type' in hourly_df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Count weather types for bar chart
                    weather_counts = hourly_df['weather_type'].value_counts().reset_index()
                    weather_counts.columns = ['Weather Type', 'Count']
                    
                    bar_chart = alt.Chart(weather_counts).mark_bar().encode(
                        x=alt.X('Count:Q', title='Hours'),
                        y=alt.Y('Weather Type:N', title='Weather Type', sort='-x'),
                        color=alt.Color('Weather Type:N', scale=alt.Scale(
                            domain=weather_counts['Weather Type'].tolist(),
                            range=[get_weather_color(wt) for wt in weather_counts['Weather Type']]
                        )),
                        tooltip=['Weather Type', 'Count']
                    ).properties(height=300)
                    st.altair_chart(bar_chart, use_container_width=True)
                
                with col2:
                    # Pie chart
                    pie_chart = alt.Chart(weather_counts).mark_arc().encode(
                        theta='Count:Q',
                        color=alt.Color('Weather Type:N', scale=alt.Scale(
                            domain=weather_counts['Weather Type'].tolist(),
                            range=[get_weather_color(wt) for wt in weather_counts['Weather Type']]
                        )),
                        tooltip=['Weather Type', 'Count']
                    ).properties(height=300)
                    st.altair_chart(pie_chart, use_container_width=True)
            
            # Summary statistics
            st.write("**Forecast Summary**")
            if 'weather_type' in hourly_df.columns:
                dominant_weather = hourly_df['weather_type'].mode().iloc[0] if not hourly_df['weather_type'].mode().empty else 'Unknown'
                avg_temp = hourly_df['temperature_2m'].mean()
                total_precip = hourly_df['precipitation'].sum()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Dominant Weather", f"{get_weather_icon(dominant_weather)} {dominant_weather}")
                col2.metric("Average Temperature", f"{avg_temp:.1f}°C")
                col3.metric("Max Temperature", f"{hourly_df['temperature_2m'].max():.1f}°C")
                col4.metric("Total Precipitation", f"{total_precip:.2f} mm")
        
        except Exception as e:
            st.error(f"Error generating hourly predictions: {e}")
            st.info("This might be due to missing historical data for lag features. Try using a different date range.")
    else:
        st.error("Models not loaded. Please run train_weather_model.py first to train the models.")

# -----------------------------
# FORECAST VIEW
# -----------------------------
elif view_option == "Forecasts":
    st.subheader("Weather Forecasts")
    if prophet_models is not None:
        feature_choice = st.selectbox(
            "Select variable to forecast:",
            ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]
        )
        
        if feature_choice in prophet_models:
            model = prophet_models[feature_choice]
            future = model.make_future_dataframe(periods=7, freq="D")
            forecast = model.predict(future)
            daily_forecast = forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(7)
            daily_forecast['ds'] = pd.to_datetime(daily_forecast['ds']).dt.date
            
            st.write("**7-Day Forecast**")
            
            # Format the table nicely
            formatted_forecast = daily_forecast.copy()
            formatted_forecast['yhat'] = formatted_forecast['yhat'].round(2)
            formatted_forecast['yhat_lower'] = formatted_forecast['yhat_lower'].round(2)
            formatted_forecast['yhat_upper'] = formatted_forecast['yhat_upper'].round(2)
            
            st.table(formatted_forecast.rename(columns={
                'ds': 'Date',
                'yhat': f'{feature_choice.replace("_", " ").title()} Forecast',
                'yhat_lower': 'Lower Bound',
                'yhat_upper': 'Upper Bound'
            }))
            
            # Add a chart for the forecast
            st.write("**Forecast Trend**")
            forecast_chart = alt.Chart(daily_forecast).mark_line(point=True).encode(
                x=alt.X('ds:T', title='Date'),
                y=alt.Y('yhat:Q', title=f'{feature_choice.replace("_", " ").title()}'),
                tooltip=['ds', 'yhat']
            ).properties(height=300)
            st.altair_chart(forecast_chart, use_container_width=True)
        else:
            st.error(f"No Prophet model available for {feature_choice}")
    else:
        st.error("Prophet models not available. Please run train_weather_model.py first.")

# -----------------------------
# FOOTER
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Data Sources:**
    - Historical data: open-meteo-27.75N85.50E1293mNew.csv
    - Forecasts: Random Forest & Prophet models trained on historical data
    - Weather types: Classified based on temperature, humidity, precipitation, and wind
    """
)

if historical_data is not None:
    st.sidebar.write(f"Historical data range: {historical_data['time'].min().date()} to {historical_data['time'].max().date()}")
    st.sidebar.write(f"Total records: {len(historical_data):,}")