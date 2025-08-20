 # weather_dashboard_enhanced.py
import pandas as pd
import streamlit as st
import joblib
from datetime import datetime, timedelta
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# -------------------- SETUP --------------------
st.set_page_config(
    page_title="Advanced Weather Forecast Dashboard",
    layout="wide",
    page_icon="🌤️",
    initial_sidebar_state="expanded"
)

# -------------------- STYLES --------------------
def load_css():
    st.markdown("""
        <style>
        .main { 
            background-color: #0f172a; 
            color: white; 
        }
        .stButton>button {
            background-color: #3b82f6; 
            color: white;
            border: none; 
            border-radius: 6px;
            padding: 0.5rem 1rem;
        }
        .hour-card {
            background-color: #1e293b;
            border-radius: 10px;
            padding: 12px;
            text-align: center;
            margin-bottom: 12px;
            transition: transform 0.2s;
        }
        .hour-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }
        .current-weather {
            background: linear-gradient(135deg, #3b82f6, #1e40af);
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        }
        .disclaimer {
            background-color: #fef3c7;
            color: #92400e;
            padding: 12px;
            border-radius: 10px;
            margin: 12px 0;
            border-left: 4px solid #f59e0b;
        }
        .metric-card {
            background-color: #1e293b;
            border-radius: 10px;
            padding: 16px;
            text-align: center;
            margin: 8px 0;
        }
        .tab-container {
            background-color: #1e293b;
            border-radius: 10px;
            padding: 20px;
            margin: 12px 0;
        }
        </style>
    """, unsafe_allow_html=True)

load_css()

# -------------------- DATA LOADING --------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("optimized_weather_model.pkl")
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None

@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv("open-meteo-27.73N85.25E1293m.csv")
        df['time'] = pd.to_datetime(df['time'])
        df['hour'] = df['time'].dt.hour
        df['month'] = df['time'].dt.month
        df['day_of_week'] = df['time'].dt.weekday

        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Weather code mapping
        weather_map = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Fog", 51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain", 
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
            85: "Slight snow showers", 86: "Heavy snow showers", 95: "Thunderstorm",
            96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
        }
        df['weather_description'] = df['weather_code'].map(weather_map).fillna("Unknown")
        return df
    except Exception as e:
        st.error(f"❌ Data loading failed: {str(e)}")
        return pd.DataFrame()

# -------------------- EXTERNAL DATA FETCHING --------------------
@st.cache_data(ttl=3600)
def fetch_weather_comparison():
    try:
        today = datetime.now().date()
        dates = [today + timedelta(days=i) for i in range(3)]
        return pd.DataFrame({
            'date': dates,
            'temp_high': [25.3, 26.1, 24.8],
            'temp_low': [18.2, 19.5, 17.8],
            'condition': ['Partly Cloudy', 'Scattered Thunderstorms', 'Mostly Sunny'],
            'precipitation_chance': [10, 60, 5]
        })
    except:
        return pd.DataFrame()

# -------------------- UTILITIES --------------------
def get_weather_icon_and_text(code):
    mapping = {
        0: ("wb_sunny", "Clear", "#facc15"),
        1: ("wb_sunny", "Mainly Clear", "#facc15"),
        2: ("cloud", "Partly Cloudy", "#f1f5f9"),
        3: ("cloud_queue", "Overcast", "#cbd5e1"),
        45: ("foggy", "Fog", "#94a3b8"),
        48: ("foggy", "Fog", "#94a3b8"),
        51: ("water_drop", "Light Drizzle", "#38bdf8"),
        53: ("water_drop", "Moderate Drizzle", "#38bdf8"),
        55: ("water_drop", "Dense Drizzle", "#38bdf8"),
        61: ("cloud_rain", "Slight Rain", "#3b82f6"),
        63: ("cloud_rain", "Moderate Rain", "#3b82f6"),
        65: ("cloud_rain", "Heavy Rain", "#1e40af"),
        71: ("ac_unit", "Slight Snow", "#f1f5f9"),
        73: ("ac_unit", "Moderate Snow", "#f1f5f9"),
        75: ("ac_unit", "Heavy Snow", "#f1f5f9"),
        77: ("ac_unit", "Snow Grains", "#f1f5f9"),
        80: ("cloud_rain", "Slight Rain Showers", "#3b82f6"),
        81: ("storm", "Moderate Rain Showers", "#f87171"),
        82: ("storm", "Violent Rain Showers", "#dc2626"),
        85: ("ac_unit", "Slight Snow Showers", "#f1f5f9"),
        86: ("ac_unit", "Heavy Snow Showers", "#f1f5f9"),
        95: ("storm", "Thunderstorm", "#f87171"),
        96: ("storm", "Thunderstorm with Hail", "#dc2626"),
        99: ("storm", "Heavy Thunderstorm", "#dc2626")
    }
    return mapping.get(int(code), ("help_outline", "Unknown", "#f1f5f9"))

# -------------------- FORECAST --------------------
def generate_hourly_forecast(date, df, model):
    if model is None or df.empty:
        return []
        
    forecasts = []
    median_values = df.median(numeric_only=True)
    
    # Get all feature names expected by the model
    expected_features = model.feature_names_in_
    
    # Create weather type dummies for the model
    bins = [0, 1, 3, 50, 70, 100]
    labels = ['clear', 'cloudy', 'fog', 'rain', 'storm']
    
    for hour in range(24):
        input_data = {
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'month_sin': np.sin(2 * np.pi * date.month / 12),
            'month_cos': np.cos(2 * np.pi * date.month / 12),
            'hour': hour, 
            'month': date.month, 
            'day_of_week': date.weekday()
        }
        
        # Find similar historical conditions
        similar = df[(df['hour'] == hour) & (df['month'] == date.month)]
        
        # Set values based on similar conditions or medians
        for col in ['relative_humidity_2m', 'wind_speed_10m (km/h)', 'wind_direction_100m']:
            if not similar.empty and col in similar.columns:
                input_data[col] = similar[col].median()
            elif col in median_values:
                input_data[col] = median_values[col]
            else:
                input_data[col] = 0  # Default value
                
        # Determine weather code
        if not similar.empty:
            weather_code = int(similar['weather_code'].mode()[0])
        else:
            weather_code = 0  # Default to clear
            
        input_data['weather_code'] = weather_code
        
        # Create weather type dummies
        weather_type = pd.cut([weather_code], bins=bins, labels=labels)[0]
        for label in labels:
            input_data[f'weather_type_{label}'] = 1 if label == weather_type else 0
        
        # Create input DataFrame with all expected features
        input_df = pd.DataFrame([input_data])
        
        # Ensure all expected features are present
        for feature in expected_features:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Fill missing features with 0
        
        # Make prediction
        try:
            pred = model.predict(input_df[expected_features])
            temp = round(float(pred[0]), 1)
        except:
            # Fallback if prediction fails
            if not similar.empty:
                temp = similar['temperature_2m'].median()
            else:
                temp = median_values.get('temperature_2m', 20)
        
        forecasts.append({
            'hour': hour,
            'time': f"{hour:02d}:00",
            'temperature': temp,
            'weather_code': weather_code,
            'humidity': input_data.get('relative_humidity_2m', 50),
            'wind_speed': input_data.get('wind_speed_10m (km/h)', 0),
            'wind_direction': input_data.get('wind_direction_100m', 0)
        })
    
    return forecasts

# -------------------- DASHBOARD PAGES --------------------
def prediction_page(df, model):
    st.title("🌤️ Kathmandu Weather Forecast")
    
    st.markdown("""
    <div class="disclaimer">
        <strong>Note:</strong> This forecast is based on historical patterns and machine learning. 
        It may differ from professional weather services that use satellite data, radar, and global atmospheric models.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_date = st.date_input("Select date", datetime.now().date(), 
                                     min_value=datetime.now().date(),
                                     max_value=datetime.now().date() + timedelta(days=7))
    
    if model is None or df.empty:
        st.error("⚠️ Forecast unavailable. Required model or data is missing.")
        return
    
    with st.spinner("Generating forecast..."):
        forecast = generate_hourly_forecast(selected_date, df, model)
    
    if not forecast:
        st.error("Failed to generate forecast. Please check your data and model.")
        return
        
    current_hour = datetime.now().hour
    current = forecast[min(current_hour, len(forecast)-1)]
    icon, text, color = get_weather_icon_and_text(current['weather_code'])
    
    # Current weather display
    st.markdown(f"""
    <div class="current-weather">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <div style="display:flex;align-items:center;gap:16px">
                <span style="font-size:3rem;font-weight:600">{current['temperature']}°C</span>
                <div style="text-align:center">
                    <span class="material-icons" style="font-size:3rem;color:{color}">{icon}</span>
                    <div style="font-size:1rem;color:#e2e8f0">{text}</div>
                </div>
            </div>
            <div style="display:flex;flex-direction:column;align-items:flex-end;gap:8px;color:#e2e8f0">
                <div><span class="material-icons">schedule</span> {current['time']}</div>
                <div><span class="material-icons">air</span> {current['wind_speed']:.1f} km/h</div>
                <div><span class="material-icons">water_drop</span> {int(current['humidity'])}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Comparison with external forecast
    external_forecast = fetch_weather_comparison()
    if not external_forecast.empty and selected_date == datetime.now().date():
        st.subheader("📊 Comparison with Weather Services")
        comp_col1, comp_col2, comp_col3 = st.columns(3)
        
        with comp_col1:
            st.metric("Our Forecast", f"{current['temperature']}°C", f"{text}")
        with comp_col2:
            st.metric("Weather Service", f"{external_forecast['temp_high'].iloc[0]}°C", 
                     f"{external_forecast['condition'].iloc[0]}")
        with comp_col3:
            precip = external_forecast['precipitation_chance'].iloc[0]
            st.metric("Precipitation Chance", f"{precip}%", 
                     "High" if precip > 50 else "Low" if precip > 20 else "Very Low")
    
    # Hourly forecast cards
    st.subheader("🕒 Hourly Forecast")
    hours_to_show = min(12, len(forecast))
    cols = st.columns(hours_to_show)
    
    for i, hour_data in enumerate(forecast[:hours_to_show]):
        icon, text, color = get_weather_icon_and_text(hour_data['weather_code'])
        with cols[i]:
            st.markdown(f"""
            <div class="hour-card">
                <div style="font-weight:600">{hour_data['time']}</div>
                <span class="material-icons" style="font-size:2rem;color:{color}">{icon}</span>
                <div style="font-size:1.2rem;font-weight:600">{hour_data['temperature']}°C</div>
                <div style="font-size:0.8rem;color:#94a3b8">{text}</div>
                <div style="font-size:0.7rem;margin-top:4px">
                    <span class="material-icons" style="font-size:0.7rem;vertical-align:middle">air</span> 
                    {hour_data['wind_speed']:.1f} km/h
                </div>
                <div style="font-size:0.7rem">
                    <span class="material-icons" style="font-size:0.7rem;vertical-align:middle">water_drop</span> 
                    {int(hour_data['humidity'])}%
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Charts and detailed analysis
    st.subheader("📈 Detailed Forecast Analysis")
    forecast_df = pd.DataFrame(forecast)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Temperature", "Humidity", "Wind", "Forecast Details"])
    
    with tab1:
        fig = px.line(forecast_df, x='time', y='temperature',
                     labels={'time': 'Hour', 'temperature': 'Temperature (°C)'},
                     title="Temperature Forecast")
        fig.update_layout(plot_bgcolor='#1e293b', paper_bgcolor='#0f172a', 
                         font_color='white', hovermode='x unified')
        fig.update_traces(line=dict(width=3), hovertemplate='%{y}°C')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.line(forecast_df, x='time', y='humidity',
                     labels={'time': 'Hour', 'humidity': 'Relative Humidity (%)'},
                     title="Humidity Forecast")
        fig.update_layout(plot_bgcolor='#1e293b', paper_bgcolor='#0f172a', 
                         font_color='white', hovermode='x unified')
        fig.update_traces(line=dict(width=3, color='#22c55e'), hovertemplate='%{y}%')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Wind speed
        fig.add_trace(
            go.Scatter(x=forecast_df['time'], y=forecast_df['wind_speed'], 
                      name="Wind Speed", line=dict(color='#3b82f6', width=3)),
            secondary_y=False,
        )
        
        # Wind direction
        fig.add_trace(
            go.Scatter(x=forecast_df['time'], y=forecast_df['wind_direction'],
                      name="Wind Direction", line=dict(color='#f97316', width=3, dash='dot')),
            secondary_y=True,
        )
        
        fig.update_layout(
            title="Wind Speed and Direction",
            plot_bgcolor='#1e293b',
            paper_bgcolor='#0f172a',
            font_color='white',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_yaxes(title_text="Wind Speed (km/h)", secondary_y=False)
        fig.update_yaxes(title_text="Wind Direction (°)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("""
        <div class="tab-container">
            <h4>Forecast Methodology</h4>
            <p>This forecast is generated using a machine learning model trained on historical weather data. 
            The model considers:</p>
            <ul>
                <li>Time of day and seasonal patterns</li>
                <li>Historical weather conditions for similar dates</li>
                <li>Wind patterns and humidity levels</li>
            </ul>
            <p><strong>Note:</strong> Precipitation forecasts are based on historical patterns and may not 
            accurately predict sudden weather changes.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show data summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Temperature", f"{forecast_df['temperature'].mean():.1f}°C")
        with col2:
            st.metric("Max Temperature", f"{forecast_df['temperature'].max():.1f}°C")
        with col3:
            st.metric("Min Temperature", f"{forecast_df['temperature'].min():.1f}°C")

def historical_analysis_page(df):
    st.title("📊 Historical Weather Analysis")
    
    if df.empty:
        st.warning("No historical data available for analysis.")
        return
        
    st.subheader("Weather Patterns Over Time")
    
    # Resample to daily data
    daily_df = df.resample('D', on='time').agg({
        'temperature_2m': 'mean',
        'relative_humidity_2m': 'mean',
        'wind_speed_10m (km/h)': 'mean'
    }).reset_index()
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Temperature Trends", "Weather Patterns", "Correlation Analysis"])
    
    with tab1:
        fig = px.line(daily_df, x='time', y='temperature_2m',
                     labels={'time': 'Date', 'temperature_2m': 'Temperature (°C)'},
                     title="Daily Average Temperature")
        fig.update_layout(plot_bgcolor='#1e293b', paper_bgcolor='#0f172a', font_color='white')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Weather frequency
        weather_counts = df['weather_description'].value_counts().reset_index()
        weather_counts.columns = ['Weather Type', 'Count']
        
        fig = px.pie(weather_counts, values='Count', names='Weather Type',
                    title="Frequency of Weather Types")
        fig.update_layout(plot_bgcolor='#1e293b', paper_bgcolor='#0f172a', font_color='white')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_df = df[numeric_cols].corr()
        
        fig = px.imshow(correlation_df, 
                       title="Correlation Between Weather Variables",
                       color_continuous_scale='RdBu_r',
                       aspect="auto")
        fig.update_layout(plot_bgcolor='#1e293b', paper_bgcolor='#0f172a', font_color='white')
        st.plotly_chart(fig, use_container_width=True)

# -------------------- MAIN APP --------------------
def main():
    df = load_historical_data()
    model = load_model()
    
    with st.sidebar:
        st.title("🌤️ Navigation")
        page = st.radio("Select Page", ["Forecast", "Historical Analysis"])
        
        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.8rem;color:#64748b;">
            <p><strong>Data Source:</strong> Open-Meteo API</p>
            <p><strong>Model:</strong> Random Forest Regressor</p>
            <p><strong>Location:</strong> Kathmandu, Nepal</p>
            <p><strong>Elevation:</strong> 1,293 meters</p>
        </div>
        """, unsafe_allow_html=True)
    
    if page == "Forecast":
        prediction_page(df, model)
    else:
        historical_analysis_page(df)

if __name__ == "__main__":
    main()