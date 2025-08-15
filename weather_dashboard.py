# weather_dashboard.py
import pandas as pd
import streamlit as st
import joblib
from datetime import datetime, timedelta
import numpy as np
import plotly.express as px
from streamlit.components.v1 import html

# -------------------- SETUP --------------------
st.set_page_config(
    page_title="Weather Forecast Dashboard",
    layout="wide",
    page_icon="⛅",
    initial_sidebar_state="expanded"
)

# -------------------- STYLES --------------------
st.markdown("""
<style>
:root {
    --primary: #0f172a;
    --secondary: #1e293b;
    --accent: #7dd3fc;
    --text: #f8fafc;
    --highlight: #0ea5e9;
}

/* Current Weather */
.current-weather {
    background: var(--secondary);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 24px;
    border-left: 4px solid var(--accent);
}

.current-main {
    display: flex;
    align-items: center;
    gap: 16px;
}

.current-temp {
    font-size: 3rem;
    font-weight: 600;
}

.current-icon {
    font-size: 3rem;
    color: var(--accent);
}

.current-details {
    display: flex;
    gap: 24px;
    margin-top: 12px;
    color: #94a3b8;
}

/* Hourly Forecast */
.hour-card {
    background: var(--secondary);
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    margin-bottom: 12px;
}

/* Material Icons */
.material-icons {
    font-family: 'Material Icons';
    font-weight: normal;
    font-style: normal;
    font-size: 24px;
    line-height: 1;
    letter-spacing: normal;
    text-transform: none;
    display: inline-block;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
    -webkit-font-feature-settings: 'liga';
    -webkit-font-smoothing: antialiased;
}
</style>
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
""", unsafe_allow_html=True)

# -------------------- DATA LOADING --------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("optimized_weather_model.pkl")
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv("open-meteo-27.73N85.25E1293m.csv")
        
        # Check required columns
        required_cols = ['time', 'temperature_2m', 'relative_humidity_2m', 
                        'wind_speed_10m (km/h)', 'weather_code']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.stop()
            
        df['time'] = pd.to_datetime(df['time'])
        
        # Feature engineering
        df['hour'] = df['time'].dt.hour
        df['month'] = df['time'].dt.month
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        # Weather categories
        bins = [0, 1, 3, 50, 70, 100]
        labels = ['clear', 'cloudy', 'fog', 'rain', 'storm']
        df['weather_type'] = pd.cut(df['weather_code'], bins=bins, labels=labels)
        df = pd.get_dummies(df, columns=['weather_type'])
        
        return df
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        st.stop()

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
    return icons.get(int(code), "help_outline")

# -------------------- FORECAST FUNCTIONS --------------------
def generate_hourly_forecast(date):
    forecasts = []
    median_values = df.median(numeric_only=True)
    
    for hour in range(24):
        # Create base input
        input_data = {
            'hour_sin': np.sin(2 * np.pi * hour/24),
            'hour_cos': np.cos(2 * np.pi * hour/24),
            'month_sin': np.sin(2 * np.pi * date.month/12),
            'month_cos': np.cos(2 * np.pi * date.month/12),
            'hour': hour,
            'month': date.month,
            'day_of_week': date.weekday()
        }
        
        # Add median values for other features
        for feature in model.feature_names_in_:
            if feature not in input_data and feature in median_values:
                input_data[feature] = median_values[feature]
        
        # Get typical weather
        similar_hour = df[df['hour'] == hour]
        if not similar_hour.empty:
            common_weather = similar_hour['weather_code'].mode()[0]
            input_data['weather_code'] = common_weather
            
            # Set weather type
            weather_type = pd.cut([common_weather], bins=[0, 1, 3, 50, 70, 100], 
                                labels=['clear', 'cloudy', 'fog', 'rain', 'storm'])[0]
            input_data[f'weather_type_{weather_type}'] = 1
        
        # Ensure all features exist
        input_df = pd.DataFrame([input_data])
        missing_features = set(model.feature_names_in_) - set(input_df.columns)
        for feature in missing_features:
            input_df[feature] = 0  # Default value
            
        # Predict temperature
        temp = round(model.predict(input_df[model.feature_names_in_])[0], 1)
        
        forecasts.append({
            'hour': hour,
            'time': f"{hour:02d}:00",
            'temperature': temp,
            'weather_code': common_weather if 'common_weather' in locals() else 0,
            'humidity': input_df['relative_humidity_2m'].values[0],
            'wind_speed': input_df['wind_speed_10m (km/h)'].values[0]
        })
    
    return forecasts

# -------------------- DASHBOARD PAGES --------------------
def prediction_page():
    st.title("Kathmandu Weather Forecast")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_date = st.date_input("Select date", datetime.now().date())
    
    forecast = generate_hourly_forecast(selected_date)
    current = forecast[datetime.now().hour]
    
    # Current weather
    st.markdown(f"""
    <div class="current-weather">
        <div class="current-main">
            <span class="current-temp">{current['temperature']}°C</span>
            <span class="material-icons">{get_weather_icon(current['weather_code'])}</span>
        </div>
        <div class="current-details">
            <div><span class="material-icons">air</span> {current['wind_speed']:.1f} km/h</div>
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
    
    tab1, tab2 = st.tabs(["Temperature", "Precipitation"])
    
    with tab1:
        st.subheader("Temperature Trends")
        temp_df = df.resample('D', on='time').mean()
        fig = px.line(temp_df, y='temperature_2m')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Precipitation Patterns")
        rain_df = df.resample('D', on='time').sum()
        fig = px.bar(rain_df, y='precipitation (mm)')
        st.plotly_chart(fig, use_container_width=True)

# -------------------- MAIN APP --------------------
def main():
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("", ["Forecast", "Historical Trends"])
        
        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.8rem; color: #64748b; margin-top: 24px;">
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