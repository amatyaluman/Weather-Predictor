import pandas as pd
import streamlit as st
import joblib
from datetime import datetime, timedelta
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------- SETUP --------------------
st.set_page_config(
    page_title="Weather Dashboard",
    layout="wide",
    page_icon="⛅",
    initial_sidebar_state="expanded"
)

# -------------------- DATA LOADING --------------------
@st.cache_resource
def load_model():
    return joblib.load("optimized_weather_model.pkl")  # Use your trained model

@st.cache_data
def load_historical_data():
    df = pd.read_csv("open-meteo-27.73N85.25E1293m.csv")
    df['time'] = pd.to_datetime(df['time'])
    
    # Feature engineering to match training
    df['hour'] = df['time'].dt.hour
    df['month'] = df['time'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # Weather code categorization
    bins = [0, 1, 3, 50, 70, 100]
    labels = ['clear', 'cloudy', 'fog', 'rain', 'storm']
    df['weather_type'] = pd.cut(df['weather_code'], bins=bins, labels=labels)
    df = pd.get_dummies(df, columns=['weather_type'])
    
    return df

model = load_model()
historical_data = load_historical_data()

# -------------------- FORECAST FUNCTIONS --------------------
def generate_hourly_forecast(requested_date):
    """Generate hourly forecast using the trained model"""
    predictions = []
    for hour in range(24):
        # Create input features
        input_features = {
            'hour_sin': np.sin(2 * np.pi * hour/24),
            'hour_cos': np.cos(2 * np.pi * hour/24),
            'month_sin': np.sin(2 * np.pi * requested_date.month/12),
            'month_cos': np.cos(2 * np.pi * requested_date.month/12),
            'relative_humidity_2m': historical_data['relative_humidity_2m'].median(),
            'wind_speed_10m (km/h)': historical_data['wind_speed_10m (km/h)'].median(),
            # Add other features with median/mean values
        }
        
        # Ensure all model features are present
        for feature in model.feature_names_in_:
            if feature not in input_features:
                input_features[feature] = 0  # Or use historical median
                
        # Make prediction
        predicted_temp = model.predict(pd.DataFrame([input_features]))[0]
        
        predictions.append({
            'hour': hour,
            'temperature': round(predicted_temp, 1),
            # Add other forecast details
        })
    return predictions

# -------------------- DASHBOARD PAGES --------------------
def show_dashboard():
    # Current weather display
    current_data = historical_data.iloc[-1]
    current_temp = round(current_data['temperature_2m'], 1)
    
    # 7-day forecast
    forecasts = []
    for i in range(7):
        date = datetime.now() + timedelta(days=i)
        forecasts.append({
            'date': date,
            'temperature': round(current_temp + np.random.uniform(-3, 3), 1),  # Replace with actual forecast
            # Add other forecast details
        })
    
    # Display metrics and charts
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(historical_data.tail(100), x='time', y='temperature_2m')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(historical_data.tail(30), x='time', y='precipitation (mm)')
        st.plotly_chart(fig, use_container_width=True)

# -------------------- MAIN APP --------------------
def main():
    pages = {
        "Dashboard": show_dashboard,
        # Add other pages
    }
    
    with st.sidebar:
        st.title("Weather Dashboard")
        page = st.radio("Navigation", list(pages.keys()))
    
    pages[page]()

if __name__ == "__main__":
    main()