import pandas as pd
import streamlit as st
import joblib
from datetime import datetime, timedelta
import numpy as np
import plotly.express as px
import os

# -------------------- SETUP --------------------
st.set_page_config(
    page_title="Weather Forecast Dashboard",
    layout="wide",
    page_icon="⛅",
    initial_sidebar_state="expanded"
)

# -------------------- STYLES --------------------
def load_css():
    if os.path.exists("styles.css"):
        with open("styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
        <style>
            body { background: #0f172a; color: white; font-family: 'Inter', sans-serif; }
            .current-weather { background: #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 24px; }
            .hour-card { background: #1e293b; border-radius: 8px; padding: 12px; margin: 8px; text-align: center; }
        </style>
        """, unsafe_allow_html=True)

load_css()

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
        required_cols = ['time', 'temperature_2m', 'relative_humidity_2m', 
                        'wind_speed_10m (km/h)', 'wind_direction_100m', 'weather_code']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
            st.stop()
            
        df['time'] = pd.to_datetime(df['time'])
        df['hour'] = df['time'].dt.hour
        df['month'] = df['time'].dt.month
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        bins = [0, 1, 3, 50, 70, 100]
        labels = ['clear', 'cloudy', 'fog', 'rain', 'storm']
        df['weather_type'] = pd.cut(df['weather_code'], bins=bins, labels=labels)
        df = pd.get_dummies(df, columns=['weather_type'])
        
        return df
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        st.stop()

model = load_model()
df = load_historical_data()

# -------------------- UTILITIES --------------------
def get_weather_icon(code):
    icons = {
        0: "wb_sunny", 1: "partly_cloudy_day", 2: "cloud", 3: "cloudy",
        45: "foggy", 48: "foggy", 51: "rainy", 53: "rainy", 55: "rainy",
        61: "rainy", 63: "rainy", 65: "rainy", 71: "ac_unit", 
        73: "ac_unit", 75: "ac_unit", 80: "rainy", 81: "thunderstorm", 
        95: "thunderstorm"
    }
    return icons.get(int(code), "help_outline")

# -------------------- FORECAST FUNCTIONS --------------------
def generate_hourly_forecast(date):
    forecasts = []
    median_values = df.median(numeric_only=True)
    
    for hour in range(24):
        input_data = {
            'hour_sin': np.sin(2 * np.pi * hour/24),
            'hour_cos': np.cos(2 * np.pi * hour/24),
            'month_sin': np.sin(2 * np.pi * date.month/12),
            'month_cos': np.cos(2 * np.pi * date.month/12),
            'hour': hour,
            'month': date.month,
            'day_of_week': date.weekday()
        }
        
        for feature in model.feature_names_in_:
            if feature not in input_data and feature in median_values:
                input_data[feature] = median_values[feature]
        
        similar_hour = df[(df['hour'] == hour) & (df['month'] == date.month)]
        if not similar_hour.empty:
            common_weather = similar_hour['weather_code'].mode()[0]
            wind_dir = similar_hour['wind_direction_100m'].median()
            input_data.update({
                'weather_code': common_weather,
                'wind_direction_100m': wind_dir
            })
            
            weather_type = pd.cut([common_weather], bins=[0, 1, 3, 50, 70, 100], 
                                labels=['clear', 'cloudy', 'fog', 'rain', 'storm'])[0]
            input_data[f'weather_type_{weather_type}'] = 1
        
        input_df = pd.DataFrame([input_data])
        for feature in set(model.feature_names_in_) - set(input_df.columns):
            input_df[feature] = 0
            
        temp = round(model.predict(input_df[model.feature_names_in_])[0], 1)
        
        forecasts.append({
            'hour': hour,
            'time': f"{hour:02d}:00",
            'temperature': temp,
            'weather_code': common_weather if 'common_weather' in locals() else 0,
            'humidity': input_df['relative_humidity_2m'].values[0],
            'wind_speed': input_df['wind_speed_10m (km/h)'].values[0],
            'wind_direction': input_df['wind_direction_100m'].values[0]
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
    
    st.markdown(f"""
    <div class="current-weather">
        <div style="display:flex;align-items:center;gap:16px">
            <span style="font-size:3rem;font-weight:600">{current['temperature']}°C</span>
            <span class="material-icons" style="font-size:3rem">{get_weather_icon(current['weather_code'])}</span>
        </div>
        <div style="display:flex;gap:24px;margin-top:12px;color:#94a3b8">
            <div><span class="material-icons">air</span> {current['wind_speed']:.1f} km/h</div>
            <div><span class="material-icons">water_drop</span> {int(current['humidity'])}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Hourly Forecast")
    cols = st.columns(8)
    for i, hour in enumerate(forecast[::3]):
        with cols[i % 8]:
            st.markdown(f"""
            <div class="hour-card">
                <div>{hour['time']}</div>
                <span class="material-icons">{get_weather_icon(hour['weather_code'])}</span>
                <div>{hour['temperature']}°C</div>
                <div style="font-size:0.8rem">
                    <span class="material-icons" style="font-size:0.8rem">air</span> {hour['wind_speed']:.1f} km/h
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Temperature", "Wind Speed", "Wind Direction"])
    
    with tab1:
        fig = px.line(pd.DataFrame(forecast), x='time', y='temperature', 
                     labels={'time': 'Hour', 'temperature': '°C'})
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.line(pd.DataFrame(forecast), x='time', y='wind_speed',
                     labels={'time': 'Hour', 'wind_speed': 'km/h'})
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        wind_df = pd.DataFrame(forecast)
        fig = px.line_polar(wind_df, r='wind_speed', theta='wind_direction',
                          line_close=True, template="plotly_dark")
        fig.update_traces(fill='toself')
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def trends_page():
    st.title("Historical Weather Trends")
    
    tab1, tab2, tab3 = st.tabs(["Temperature", "Precipitation", "Wind"])
    
    with tab1:
        st.subheader("Temperature Trends")
        fig = px.line(df.resample('D', on='time').mean(), y='temperature_2m')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Precipitation Patterns")
        fig = px.bar(df.resample('D', on='time').sum(), y='precipitation (mm)')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Wind Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Wind Speed Distribution**")
            fig = px.histogram(df, x='wind_speed_10m (km/h)')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("**Wind Direction Rose**")
            fig = px.bar_polar(df, r="wind_speed_10m (km/h)", 
                              theta="wind_direction_100m",
                              color="wind_speed_10m (km/h)",
                              template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

# -------------------- MAIN APP --------------------
def main():
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("", ["Forecast", "Historical Trends"])
        
        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.8rem;color:#64748b;margin-top:24px">
            <div>Data Source: Open-Meteo API</div>
            <div>Model: Random Forest</div>
            <div>Location: Kathmandu, Nepal</div>
        </div>
        """, unsafe_allow_html=True)
    
    if page == "Forecast":
        prediction_page()
    else:
        trends_page()

if __name__ == "__main__":
    main()