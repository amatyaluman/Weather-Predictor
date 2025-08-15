# weather_dashboard.py
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
    page_title="Weather Forecast Dashboard",
    layout="wide",
    page_icon="⛅",
    initial_sidebar_state="expanded"
)

# -------------------- STYLES --------------------
with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# -------------------- DATA LOADING --------------------
@st.cache_resource
def load_model():
    return joblib.load("optimized_weather_model.pkl")

@st.cache_data
def load_historical_data():
    df = pd.read_csv("open-meteo-27.73N85.25E1293m.csv")
    df['time'] = pd.to_datetime(df['time'])
    
    # Feature engineering to match model training
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
df = load_historical_data()

# -------------------- UTILITIES --------------------
def get_weather_icon(code):
    icons = {
        0: "wb_sunny",
        1: "partly_cloudy_day",
        2: "cloud",
        3: "cloudy",
        45: "foggy",
        48: "foggy",
        51: "rainy",
        53: "rainy",
        55: "rainy",
        61: "rainy",
        63: "rainy",
        65: "rainy",
        71: "ac_unit",
        73: "ac_unit",
        75: "ac_unit",
        80: "rainy",
        81: "thunderstorm",
        95: "thunderstorm"
    }
    return icons.get(code, "help_outline")

# -------------------- FORECAST FUNCTIONS --------------------
def generate_hourly_forecast(date):
    """Generate hourly forecast using the trained model"""
    forecasts = []
    for hour in range(24):
        # Create input features matching model training
        input_data = {
            'hour_sin': np.sin(2 * np.pi * hour/24),
            'hour_cos': np.cos(2 * np.pi * hour/24),
            'month_sin': np.sin(2 * np.pi * date.month/12),
            'month_cos': np.cos(2 * np.pi * date.month/12),
            'relative_humidity_2m': df['relative_humidity_2m'].median(),
            'wind_speed_10m (km/h)': df['wind_speed_10m (km/h)'].median(),
            'wind_speed_100m (km/h)': df['wind_speed_100m (km/h)'].median(),
            'wind_direction_100m': df['wind_direction_100m'].median(),
            'precipitation (mm)': 0,
            'rain (mm)': 0,
            'weather_type_clear': 0,
            'weather_type_cloudy': 0,
            'weather_type_fog': 0,
            'weather_type_rain': 0,
            'weather_type_storm': 0
        }
        
        # Set weather type based on historical patterns
        similar_hour = df[df['hour'] == hour]
        if not similar_hour.empty:
            common_weather = similar_hour['weather_code'].mode()[0]
            weather_type = pd.cut([common_weather], bins=[0, 1, 3, 50, 70, 100], 
                                labels=['clear', 'cloudy', 'fog', 'rain', 'storm'])[0]
            input_data[f'weather_type_{weather_type}'] = 1
        
        # Make prediction
        input_df = pd.DataFrame([input_data])[model.feature_names_in_]
        temp = round(model.predict(input_df)[0], 1)
        
        forecasts.append({
            'hour': hour,
            'time': f"{hour:02d}:00",
            'temperature': temp,
            'weather_code': common_weather if 'common_weather' in locals() else 0,
            'humidity': input_data['relative_humidity_2m'],
            'wind_speed': input_data['wind_speed_10m (km/h)']
        })
    return forecasts

# -------------------- DASHBOARD PAGES --------------------
def prediction_page():
    st.title("Kathmandu Weather Forecast")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_date = st.date_input("Select date", datetime.now().date())
    
    with st.spinner("Generating forecast..."):
        forecast = generate_hourly_forecast(selected_date)
    
    # Current weather card
    current = forecast[datetime.now().hour]
    st.markdown(f"""
    <div class="current-weather">
        <div class="current-main">
            <span class="current-temp">{current['temperature']}°C</span>
            <span class="material-icons current-icon">{get_weather_icon(current['weather_code'])}</span>
        </div>
        <div class="current-details">
            <div><span class="material-icons">air</span> {current['wind_speed']} km/h</div>
            <div><span class="material-icons">water_drop</span> {int(current['humidity'])}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hourly forecast
    st.subheader("Hourly Forecast")
    cols = st.columns(8)
    for i, hour in enumerate(forecast[::3]):  # Show every 3 hours
        with cols[i % 8]:
            st.markdown(f"""
            <div class="hour-card">
                <div>{hour['time']}</div>
                <span class="material-icons">{get_weather_icon(hour['weather_code'])}</span>
                <div>{hour['temperature']}°C</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Temperature chart
    fig = px.line(
        pd.DataFrame(forecast), 
        x='time', y='temperature',
        title="Temperature Trend",
        labels={'time': 'Hour', 'temperature': '°C'}
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color="#ffffff"
    )
    st.plotly_chart(fig, use_container_width=True)

def trends_page():
    st.title("Historical Weather Trends")
    
    tab1, tab2, tab3 = st.tabs(["Temperature", "Precipitation", "Wind"])
    
    with tab1:
        st.subheader("Temperature Trends")
        fig = px.line(
            df.resample('D', on='time').mean(),
            y='temperature_2m',
            labels={'temperature_2m': 'Temperature (°C)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Precipitation Patterns")
        fig = px.bar(
            df.resample('D', on='time').sum(),
            y='precipitation (mm)',
            labels={'precipitation (mm)': 'Precipitation (mm)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Wind Speed Distribution")
        fig = px.histogram(
            df, 
            x='wind_speed_10m (km/h)',
            nbins=20,
            labels={'wind_speed_10m (km/h)': 'Wind Speed (km/h)'}
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------- MAIN APP --------------------
def main():
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("", ["Forecast", "Historical Trends"])
        
        st.markdown("---")
        st.markdown("""
        <div class="sidebar-footer">
            <div>Data Source: Open-Meteo API</div>
            <div>Model: Random Forest</div>
        </div>
        """, unsafe_allow_html=True)
    
    if page == "Forecast":
        prediction_page()
    else:
        trends_page()

if __name__ == "__main__":
    main()