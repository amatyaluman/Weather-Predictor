import pandas as pd
import streamlit as st
import joblib
from datetime import datetime
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
    st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
        <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    """, unsafe_allow_html=True)

    if os.path.exists("styles.css"):
        with open("styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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
        required_cols = [
            'time', 'temperature_2m', 'relative_humidity_2m', 
            'wind_speed_10m (km/h)', 'wind_direction_100m', 'weather_code'
        ]
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
def get_weather_icon_and_text(code):
    mapping = {
        0: ("wb_sunny", "Clear", "#facc15"),
        1: ("wb_sunny", "Clear", "#facc15"),
        2: ("cloud", "Partly Cloudy", "#f1f5f9"),
        3: ("cloud_queue", "Cloudy", "#f1f5f9"),
        45: ("foggy", "Fog", "#94a3b8"),
        48: ("foggy", "Fog", "#94a3b8"),
        51: ("water_drop", "Drizzle", "#38bdf8"),
        53: ("water_drop", "Drizzle", "#38bdf8"),
        55: ("water_drop", "Drizzle", "#38bdf8"),
        61: ("cloud_rain", "Rain", "#3b82f6"),
        63: ("cloud_rain", "Rain", "#3b82f6"),
        65: ("cloud_rain", "Heavy Rain", "#1e40af"),
        71: ("ac_unit", "Snow", "#f1f5f9"),
        73: ("ac_unit", "Snow", "#f1f5f9"),
        75: ("ac_unit", "Snow", "#f1f5f9"),
        80: ("cloud_rain", "Rain Showers", "#3b82f6"),
        81: ("storm", "Thunderstorm", "#f87171"),
        95: ("storm", "Thunderstorm", "#f87171")
    }
    return mapping.get(int(code), ("help_outline", "Unknown", "#f1f5f9"))

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
            wind_speed_val = similar_hour['wind_speed_10m (km/h)'].median()
            
            input_data.update({
                'weather_code': common_weather,
                'wind_direction_100m': wind_dir,
                'wind_speed_10m (km/h)': wind_speed_val
            })
            
            weather_type = pd.cut([common_weather], bins=[0, 1, 3, 50, 70, 100], 
                                labels=['clear', 'cloudy', 'fog', 'rain', 'storm'])[0]
            input_data[f'weather_type_{weather_type}'] = 1
        else:
            wind_speed_val = median_values['wind_speed_10m (km/h)']
        
        input_df = pd.DataFrame([input_data])
        for feature in set(model.feature_names_in_) - set(input_df.columns):
            input_df[feature] = 0
            
        pred = model.predict(input_df[model.feature_names_in_])
        temp = round(float(pred[0, 0]) if pred.ndim > 1 else float(pred[0]), 1)
        
        humidity = float(input_df.get('relative_humidity_2m', [median_values['relative_humidity_2m']])[0])
        wind_direction = float(input_df.get('wind_direction_100m', [median_values['wind_direction_100m']])[0])
        
        forecasts.append({
            'hour': hour,
            'time': f"{hour:02d}:00",
            'temperature': temp,
            'weather_code': common_weather if 'common_weather' in locals() else 0,
            'humidity': humidity,
            'wind_speed': wind_speed_val,
            'wind_direction': wind_direction
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
    icon, text, color = get_weather_icon_and_text(current['weather_code'])
    
    st.markdown(f"""
    <div class="current-weather">
        <div style="display:flex;align-items:center;gap:16px">
            <span style="font-size:3rem;font-weight:600">{current['temperature']}°C</span>
            <div style="text-align:center">
                <span class="material-icons" style="font-size:3rem;color:{color}">{icon}</span>
                <div style="font-size:1rem;color:#94a3b8">{text}</div>
            </div>
        </div>
        <div style="display:flex;gap:24px;margin-top:12px;color:#94a3b8">
            <div><span class="material-icons">air</span> {current['wind_speed']:.1f} km/h</div>
            <div><span class="material-icons">water_drop</span> {int(current['humidity'])}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Hourly Forecast")
    cols = st.columns(8)
    for i, hour in enumerate(forecast): 
        icon, text, color = get_weather_icon_and_text(hour['weather_code'])
        with cols[i % 8]:
            st.markdown(f"""
            <div class="hour-card">
                <div>{hour['time']}</div>
                <span class="material-icons" style="color:{color}">{icon}</span>
                <div>{hour['temperature']}°C</div>
                <div style="font-size:0.8rem;color:#94a3b8">{text}</div>
                <div style="font-size:0.8rem">
                    <span class="material-icons" style="font-size:0.8rem">air</span> {hour['wind_speed']:.1f} km/h
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Temperature", "Wind Speed", "Wind Direction"])
    
    with tab1:
        fig = px.line(pd.DataFrame(forecast), x='time', y='temperature', 
                     labels={'time': 'Hour', 'temperature': '°C'})
        fig.update_layout(plot_bgcolor='#0f172a', paper_bgcolor='#0f172a', font_color='white')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.line(pd.DataFrame(forecast), x='time', y='wind_speed',
                     labels={'time': 'Hour', 'wind_speed': 'km/h'})
        fig.update_layout(plot_bgcolor='#0f172a', paper_bgcolor='#0f172a', font_color='white')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        wind_df = pd.DataFrame(forecast)
        fig = px.bar_polar(
            wind_df, r='wind_speed', theta='wind_direction',
            color='wind_speed', color_continuous_scale=px.colors.sequential.Blues
        )
        fig.update_layout(
            paper_bgcolor='#0f172a',
            plot_bgcolor='#0f172a',
            font_color='white',
            polar=dict(
                bgcolor='#0f172a',
                angularaxis=dict(gridcolor='gray', linecolor='white'),
                radialaxis=dict(gridcolor='gray', linecolor='white', visible=True)
            ),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

def trends_page():
    st.title("Historical Weather Trends")
    
    tab1, tab2, tab3 = st.tabs(["Temperature", "Precipitation", "Wind"])
    
    with tab1:
        st.subheader("Temperature Trends")
        fig = px.line(df.resample('D', on='time').mean(), y='temperature_2m')
        fig.update_layout(plot_bgcolor='#0f172a', paper_bgcolor='#0f172a', font_color='white')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Precipitation Patterns")
        fig = px.bar(df.resample('D', on='time').sum(), y='precipitation (mm)')
        fig.update_layout(plot_bgcolor='#0f172a', paper_bgcolor='#0f172a', font_color='white')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Wind Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Wind Speed Distribution**")
            fig = px.histogram(df, x='wind_speed_10m (km/h)')
            fig.update_layout(plot_bgcolor='#0f172a', paper_bgcolor='#0f172a', font_color='white')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("**Wind Direction Rose**")
            fig = px.bar_polar(
                df, r="wind_speed_10m (km/h)", 
                theta="wind_direction_100m",
                color="wind_speed_10m (km/h)",
                color_continuous_scale=px.colors.sequential.Blues
            )
            fig.update_layout(
                paper_bgcolor='#0f172a',
                plot_bgcolor='#0f172a',
                font_color='white',
                polar=dict(
                    bgcolor='#0f172a',
                    angularaxis=dict(gridcolor='gray', linecolor='white'),
                    radialaxis=dict(gridcolor='gray', linecolor='white', visible=True)
                ),
                showlegend=False
            )
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
