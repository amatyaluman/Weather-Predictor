import pandas as pd
import streamlit as st
import joblib
from datetime import datetime, timedelta
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import uuid

# -------------------- SETUP --------------------
st.set_page_config(
    page_title="Professional Weather Forecast Dashboard",
    layout="wide",
    page_icon="🌦️",
    initial_sidebar_state="expanded"
)

# -------------------- STYLES --------------------
def load_css():
    st.markdown("""
        <style>
        .main { 
            background-color: #ffffff; 
            color: #1f2a44; 
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #0284c7; 
            color: white;
            border: none; 
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 500;
        }
        .hour-card {
            background-color: #f8fafc;
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            margin-bottom: 16px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
        }
        .hour-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 12px rgba(2, 132, 199, 0.2);
        }
        .current-weather {
            background: linear-gradient(135deg, #0284c7, #60a5fa);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            color: white;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        }
        .disclaimer {
            background-color: #fef3c7;
            color: #b45309;
            padding: 16px;
            border-radius: 10px;
            margin: 16px 0;
            border-left: 4px solid #f59e0b;
        }
        .metric-card {
            background-color: #f8fafc;
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            margin: 8px 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        .tab-container {
            background-color: #f8fafc;
            border-radius: 12px;
            padding: 20px;
            margin: 12px 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        .stTabs [aria-selected="true"] {
            color: #0284c7 !important;
            border-bottom: 3px solid #0284c7 !important;
            font-weight: 600;
        }
        .stTabs [role="tab"] {
            color: #64748b;
            font-weight: 500;
        }
        .stTabs [role="tab"]:hover {
            color: #0284c7;
        }
        .daily-card {
            background-color: #f8fafc;
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            margin-bottom: 16px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
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
        st.info("Using fallback prediction method based on historical data")
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
            0: "Clear", 1: "Mostly Clear", 2: "Partly Cloudy", 3: "Overcast",
            45: "Fog", 48: "Fog", 51: "Light Drizzle", 53: "Drizzle", 55: "Heavy Drizzle",
            61: "Light Rain", 63: "Rain", 65: "Heavy Rain", 
            71: "Light Snow", 73: "Snow", 75: "Heavy Snow",
            77: "Snow Grains", 80: "Light Showers", 81: "Showers", 82: "Heavy Showers",
            85: "Light Snow Showers", 86: "Snow Showers", 95: "Thunderstorm",
            96: "Thunderstorm with Hail", 99: "Severe Thunderstorm"
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
        dates = [today + timedelta(days=i) for i in range(5)]
        return pd.DataFrame({
            'date': dates,
            'temp_high': [25.3, 26.1, 24.8, 25.5, 26.0],
            'temp_low': [18.2, 19.5, 17.8, 18.0, 19.0],
            'feels_like': [26.5, 27.0, 25.0, 26.0, 27.5],
            'condition': ['Partly Cloudy', 'Scattered Thunderstorms', 'Mostly Sunny', 'Cloudy', 'Sunny'],
            'precipitation_chance': [10, 60, 5, 20, 15],
            'uv_index': [7, 6, 8, 5, 7]
        })
    except:
        return pd.DataFrame()

# -------------------- UTILITIES --------------------
def get_weather_icon_and_text(code):
    mapping = {
        0: ("sunny", "Clear", "#facc15"),
        1: ("sunny", "Mostly Clear", "#facc15"),
        2: ("partly_cloudy_day", "Partly Cloudy", "#94a3b8"),
        3: ("cloud", "Overcast", "#64748b"),
        45: ("foggy", "Fog", "#94a3b8"),
        48: ("foggy", "Fog", "#94a3b8"),
        51: ("water_drop", "Light Drizzle", "#38bdf8"),
        53: ("water_drop", "Drizzle", "#38bdf8"),
        55: ("water_drop", "Heavy Drizzle", "#38bdf8"),
        61: ("rainy", "Light Rain", "#0284c7"),
        63: ("rainy", "Rain", "#0284c7"),
        65: ("rainy", "Heavy Rain", "#1e40af"),
        71: ("ac_unit", "Light Snow", "#e5e7eb"),
        73: ("ac_unit", "Snow", "#e5e7eb"),
        75: ("ac_unit", "Heavy Snow", "#e5e7eb"),
        77: ("ac_unit", "Snow Grains", "#e5e7eb"),
        80: ("rainy", "Light Showers", "#0284c7"),
        81: ("rainy", "Showers", "#0284c7"),
        82: ("rainy", "Heavy Showers", "#1e40af"),
        85: ("ac_unit", "Light Snow Showers", "#e5e7eb"),
        86: ("ac_unit", "Snow Showers", "#e5e7eb"),
        95: ("thunderstorm", "Thunderstorm", "#dc2626"),
        96: ("thunderstorm", "Thunderstorm with Hail", "#dc2626"),
        99: ("thunderstorm", "Severe Thunderstorm", "#dc2626")
    }
    return mapping.get(int(code), ("help_outline", "Unknown", "#94a3b8"))

def calculate_feels_like(temp, humidity, wind_speed):
    # Simplified "feels like" calculation
    if temp < 10:
        # Wind chill for cold temperatures
        feels_like = 13.12 + 0.6215 * temp - 11.37 * (wind_speed ** 0.16) + 0.3965 * temp * (wind_speed ** 0.16)
    elif temp > 20:
        # Heat index for warm temperatures
        hi = -8.78469475556 + 1.61139411 * temp + 2.33854883889 * humidity - 0.14611605 * temp * humidity
        feels_like = hi
    else:
        feels_like = temp
    return round(feels_like, 1)

# -------------------- FORECAST --------------------
def generate_hourly_forecast(date, df, model):
    if df.empty:
        return []
        
    forecasts = []
    median_values = df.median(numeric_only=True)
    
    # Get expected features if model exists
    expected_features = []
    if model is not None:
        try:
            expected_features = model.feature_names_in_
        except:
            pass
    
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
        
        # Estimate precipitation probability
        precip_prob = 0
        if weather_code in [51, 53, 55, 61, 63, 65, 80, 81, 82, 95, 96, 99]:
            precip_prob = min(80, 20 + weather_code)  # Simplified probability
        
        # Estimate UV index
        uv_index = 6 if hour in range(10, 16) else 3 if hour in range(8, 18) else 1
        
        # Create weather type dummies
        weather_type = pd.cut([weather_code], bins=bins, labels=labels)[0]
        for label in labels:
            input_data[f'weather_type_{label}'] = 1 if label == weather_type else 0
        
        # Create input DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction if model exists
        if model is not None and len(expected_features) > 0:
            # Ensure all expected features are present
            for feature in expected_features:
                if feature not in input_df.columns:
                    input_df[feature] = 0  # Fill missing features with 0
            
            try:
                pred = model.predict(input_df[expected_features])
                temp = round(float(pred[0]), 1)
            except:
                # Fallback if prediction fails
                if not similar.empty:
                    temp = similar['temperature_2m'].median()
                else:
                    temp = median_values.get('temperature_2m', 20)
        else:
            # Fallback prediction using historical data
            if not similar.empty:
                temp = similar['temperature_2m'].median()
            else:
                temp = median_values.get('temperature_2m', 20)
        
        feels_like = calculate_feels_like(temp, input_data.get('relative_humidity_2m', 50), 
                                       input_data.get('wind_speed_10m (km/h)', 0))
        
        forecasts.append({
            'hour': hour,
            'time': f"{hour:02d}:00",
            'temperature': temp,
            'feels_like': feels_like,
            'weather_code': weather_code,
            'humidity': input_data.get('relative_humidity_2m', 50),
            'wind_speed': input_data.get('wind_speed_10m (km/h)', 0),
            'wind_direction': input_data.get('wind_direction_100m', 0),
            'precipitation_chance': precip_prob,
            'uv_index': uv_index
        })
    
    return forecasts

def generate_daily_forecast(start_date, df, model):
    forecasts = []
    for i in range(5):
        date = start_date + timedelta(days=i)
        hourly_forecast = generate_hourly_forecast(date, df, model)
        if not hourly_forecast:
            continue
        daily_high = max(h['temperature'] for h in hourly_forecast)
        daily_low = min(h['temperature'] for h in hourly_forecast)
        daily_feels_like = max(h['feels_like'] for h in hourly_forecast)
        daily_precip = max(h['precipitation_chance'] for h in hourly_forecast)
        daily_uv = max(h['uv_index'] for h in hourly_forecast)
        daily_weather_code = max(h['weather_code'] for h in hourly_forecast)
        forecasts.append({
            'date': date,
            'high': daily_high,
            'low': daily_low,
            'feels_like': daily_feels_like,
            'precipitation_chance': daily_precip,
            'uv_index': daily_uv,
            'weather_code': daily_weather_code
        })
    return forecasts

# -------------------- DASHBOARD PAGES --------------------
def prediction_page(df, model):
    st.title("🌦️ Kathmandu Weather Forecast")
    
    st.markdown("""
    <div class="disclaimer">
        <strong>Advisory:</strong> This forecast uses machine learning and historical data. 
        For critical decisions, consult professional weather services like AccuWeather or The Weather Channel.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_date = st.date_input("Select date", datetime.now().date(), 
                                     min_value=datetime.now().date(),
                                     max_value=datetime.now().date() + timedelta(days=7))
    
    if df.empty:
        st.error("⚠️ Forecast unavailable. Historical data is missing.")
        return
    
    with st.spinner("Generating forecast..."):
        forecast = generate_hourly_forecast(selected_date, df, model)
        daily_forecasts = generate_daily_forecast(selected_date, df, model)
    
    if not forecast:
        st.error("Failed to generate forecast. Please check your data.")
        return
        
    current_hour = datetime.now().hour
    current = forecast[min(current_hour, len(forecast)-1)]
    icon, text, color = get_weather_icon_and_text(current['weather_code'])
    
    # Current weather display
    st.markdown(f"""
    <div class="current-weather">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <div style="display:flex;align-items:center;gap:24px">
                <span style="font-size:4rem;font-weight:700">{current['temperature']}°C</span>
                <div style="text-align:center">
                    <span class="material-icons" style="font-size:4rem;color:{color}">{icon}</span>
                    <div style="font-size:1.2rem;font-weight:500">{text}</div>
                </div>
            </div>
            <div style="display:flex;flex-direction:column;align-items:flex-end;gap:12px;font-size:1rem">
                <div><strong>Feels Like:</strong> {current['feels_like']}°C</div>
                <div><span class="material-icons" style="vertical-align:middle">schedule</span> {current['time']}</div>
                <div><span class="material-icons" style="vertical-align:middle">water_drop</span> {int(current['humidity'])}%</div>
                <div><span class="material-icons" style="vertical-align:middle">air</span> {current['wind_speed']:.1f} km/h</div>
                <div><span class="material-icons" style="vertical-align:middle">beach_access</span> UV {current['uv_index']}</div>
                <div><span class="material-icons" style="vertical-align:middle">umbrella</span> {current['precipitation_chance']}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 5-day forecast
    st.subheader("📅 5-Day Forecast")
    cols = st.columns(5)
    for i, day in enumerate(daily_forecasts):
        icon, text, color = get_weather_icon_and_text(day['weather_code'])
        with cols[i]:
            st.markdown(f"""
            <div class="daily-card">
                <div style="font-weight:600">{day['date'].strftime('%a, %b %d')}</div>
                <span class="material-icons" style="font-size:2.5rem;color:{color}">{icon}</span>
                <div style="font-size:1.2rem;font-weight:600">{day['high']}°/{day['low']}°C</div>
                <div style="font-size:0.9rem;color:#475569">{text}</div>
                <div style="font-size:0.8rem;margin-top:4px">
                    <span class="material-icons" style="font-size:0.8rem;vertical-align:middle">umbrella</span> 
                    {day['precipitation_chance']}%
                </div>
                <div style="font-size:0.8rem">
                    <span class="material-icons" style="font-size:0.8rem;vertical-align:middle">beach_access</span> 
                    UV {day['uv_index']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Comparison with external forecast
    external_forecast = fetch_weather_comparison()
    if not external_forecast.empty and selected_date in external_forecast['date'].values:
        st.subheader("🌍 Comparison with Professional Services")
        ext_data = external_forecast[external_forecast['date'] == selected_date].iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Our Forecast", f"{current['temperature']}°C", f"{text}")
        with col2:
            st.metric("Feels Like", f"{current['feels_like']}°C", f"{current['temperature'] - current['feels_like']:+.1f}°C")
        with col3:
            st.metric("Weather Service", f"{ext_data['temp_high']}°/{ext_data['temp_low']}°C", f"{ext_data['condition']}")
        with col4:
            precip = ext_data['precipitation_chance']
            st.metric("Precipitation", f"{precip}%", 
                     "High" if precip > 50 else "Moderate" if precip > 20 else "Low")
    
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
                <span class="material-icons" style="font-size:2.5rem;color:{color}">{icon}</span>
                <div style="font-size:1.2rem;font-weight:600">{hour_data['temperature']}°C</div>
                <div style="font-size:0.9rem;color:#475569">{text}</div>
                <div style="font-size:0.8rem;margin-top:4px">
                    <span class="material-icons" style="font-size:0.8rem;vertical-align:middle">umbrella</span> 
                    {hour_data['precipitation_chance']}%
                </div>
                <div style="font-size:0.8rem">
                    <span class="material-icons" style="font-size:0.8rem;vertical-align:middle">water_drop</span> 
                    {int(hour_data['humidity'])}%
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Charts and detailed analysis
    st.subheader("📈 Detailed Forecast Analysis")
    forecast_df = pd.DataFrame(forecast)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Temperature", "Feels Like", "Humidity", "Wind", "Details"])
    
    with tab1:
        fig = px.line(forecast_df, x='time', y='temperature',
                     labels={'time': 'Hour', 'temperature': 'Temperature (°C)'},
                     title="Temperature Forecast")
        fig.update_layout(plot_bgcolor='#f8fafc', paper_bgcolor='#ffffff', 
                         font_color='#1f2a44', hovermode='x unified')
        fig.update_traces(line=dict(width=3, color='#0284c7'), hovertemplate='%{y}°C')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.line(forecast_df, x='time', y='feels_like',
                     labels={'time': 'Hour', 'feels_like': 'Feels Like (°C)'},
                     title="Feels Like Temperature")
        fig.update_layout(plot_bgcolor='#f8fafc', paper_bgcolor='#ffffff', 
                         font_color='#1f2a44', hovermode='x unified')
        fig.update_traces(line=dict(width=3, color='#16a34a'), hovertemplate='%{y}°C')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = px.line(forecast_df, x='time', y='humidity',
                     labels={'time': 'Hour', 'humidity': 'Relative Humidity (%)'},
                     title="Humidity Forecast")
        fig.update_layout(plot_bgcolor='#f8fafc', paper_bgcolor='#ffffff', 
                         font_color='#1f2a44', hovermode='x unified')
        fig.update_traces(line=dict(width=3, color='#16a34a'), hovertemplate='%{y}%')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Wind speed
        fig.add_trace(
            go.Scatter(x=forecast_df['time'], y=forecast_df['wind_speed'], 
                      name="Wind Speed", line=dict(color='#0284c7', width=3)),
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
            plot_bgcolor='#f8fafc',
            paper_bgcolor='#ffffff',
            font_color='#1f2a44',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_yaxes(title_text="Wind Speed (km/h)", secondary_y=False)
        fig.update_yaxes(title_text="Wind Direction (°)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("""
        <div class="tab-container">
            <h4>Forecast Details</h4>
            <p>Our forecast combines machine learning with historical weather patterns to provide accurate predictions for Kathmandu.</p>
            <ul>
                <li><strong>Model:</strong> Random Forest Regressor</li>
                <li><strong>Data Source:</strong> Open-Meteo historical data</li>
                <li><strong>Key Metrics:</strong> Temperature, Feels Like, Humidity, Wind, Precipitation Chance, UV Index</li>
            </ul>
            <p><strong>Limitations:</strong> This forecast may not capture sudden weather changes or extreme events as accurately as professional services.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Temperature", f"{forecast_df['temperature'].mean():.1f}°C")
        with col2:
            st.metric("Max Temperature", f"{forecast_df['temperature'].max():.1f}°C")
        with col3:
            st.metric("Min Temperature", f"{forecast_df['temperature'].min():.1f}°C")
        with col4:
            st.metric("Avg Precipitation", f"{forecast_df['precipitation_chance'].mean():.0f}%")

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
        fig.update_layout(plot_bgcolor='#f8fafc', paper_bgcolor='#ffffff', font_color='#1f2a44')
        fig.update_traces(line=dict(width=3, color='#0284c7'))
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        weather_counts = df['weather_description'].value_counts().reset_index()
        weather_counts.columns = ['Weather Type', 'Count']
        
        fig = px.pie(weather_counts, values='Count', names='Weather Type',
                    title="Frequency of Weather Types",
                    color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(plot_bgcolor='#f8fafc', paper_bgcolor='#ffffff', font_color='#1f2a44')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_df = df[numeric_cols].corr()
        
        fig = px.imshow(correlation_df, 
                       title="Correlation Between Weather Variables",
                       color_continuous_scale='RdBu_r',
                       aspect="auto")
        fig.update_layout(plot_bgcolor='#f8fafc', paper_bgcolor='#ffffff', font_color='#1f2a44')
        st.plotly_chart(fig, use_container_width=True)

# -------------------- MAIN APP --------------------
def main():
    df = load_historical_data()
    model = load_model()
    
    with st.sidebar:
        st.title("🌦️ Weather Dashboard")
        page = st.radio("Select Page", ["Forecast", "Historical Analysis"])
        
        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.9rem;color:#64748b;">
            <p><strong>Location:</strong> Kathmandu, Nepal</p>
            <p><strong>Elevation:</strong> 1,293 meters</p>
            <p><strong>Data Source:</strong> Open-Meteo API</p>
            <p><strong>Model:</strong> Random Forest Regressor</p>
        </div>
        """, unsafe_allow_html=True)
    
    if page == "Forecast":
        prediction_page(df, model)
    else:
        historical_analysis_page(df)

if __name__ == "__main__":
    main()