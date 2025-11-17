# weather_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Kathmandu Weather Forecasting System",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session State
if 'forecast_generated' not in st.session_state:
    st.session_state.forecast_generated = False
if 'current_forecast' not in st.session_state:
    st.session_state.current_forecast = None
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None

# ==================== ENHANCED CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); font-family: 'Inter', sans-serif; color: #1e293b;}
    
    .header-title {
        font-size: 3rem; 
        font-weight: 800; 
        text-align: center; 
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 1.5rem 0 0.5rem 0;
        padding: 0.5rem;
    }
    
    .header-subtitle {
        text-align: center; 
        color: #64748b; 
        font-size: 1.2rem; 
        font-weight: 400;
        margin-bottom: 2.5rem;
    }
    
    .section-header {
        font-size: 1.8rem; 
        font-weight: 700; 
        color: #1e40af; 
        margin: 2.5rem 0 1.5rem 0; 
        padding-bottom: 0.7rem; 
        border-bottom: 3px solid #3b82f6;
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.8rem; 
        border-radius: 16px; 
        box-shadow: 0 8px 25px rgba(0,0,0,0.1); 
        text-align: center; 
        border: 1px solid #e2e8f0; 
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
    }
    
    .metric-card:hover {
        transform: translateY(-8px); 
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2.6rem; 
        font-weight: 800; 
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #64748b; 
        font-size: 1.1rem; 
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .forecast-card {
        background: white; 
        padding: 1.4rem; 
        border-radius: 14px; 
        box-shadow: 0 6px 18px rgba(0,0,0,0.08); 
        border-left: 6px solid #3b82f6; 
        margin: 0.8rem 0;
        transition: all 0.3s ease;
    }
    
    .forecast-card:hover {
        border-left-color: #1d4ed8; 
        transform: translateX(8px);
        box-shadow: 0 8px 22px rgba(0,0,0,0.12);
    }
    
    .condition-badge {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8); 
        color: white; 
        padding: 0.5rem 1.2rem; 
        border-radius: 50px; 
        font-weight: 600; 
        font-size: 0.9rem;
        display: inline-block;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white; 
        border: none; 
        padding: 0.9rem 2rem; 
        border-radius: 12px; 
        font-weight: 600; 
        width: 100%; 
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: white;
    }
    
    .plotly-graph-div {
        border-radius: 16px; 
        overflow: hidden; 
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .weather-alert {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        border-left: 6px solid #f59e0b;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .data-source {
        font-size: 0.85rem;
        color: #64748b;
        text-align: center;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== ENHANCED CORE CLASSES ====================
class AdvancedFeatureEngineer:
    def __init__(self):
        self.required_features = []
        
    def add_temporal_features(self, df):
        df['hour'] = df['time'].dt.hour
        df['month'] = df['time'].dt.month
        df['day_of_year'] = df['time'].dt.dayofyear
        df['day_of_week'] = df['time'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        return df

    def add_lagged_features(self, df, lags=[1, 3, 6, 12, 24]):
        cols = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 
                'precipitation', 'pressure_msl', 'cloud_cover']
        
        for col in [c for c in cols if c in df.columns]:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                
        return df

    def add_rolling_features(self, df, windows=[3, 6, 12, 24]):
        cols = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 
                'pressure_msl', 'cloud_cover']
        
        for col in [c for c in cols if c in df.columns]:
            s = df[col].shift(1)
            for w in windows:
                df[f'{col}_rolling_mean_{w}'] = s.rolling(w, min_periods=1).mean()
                df[f'{col}_rolling_std_{w}'] = s.rolling(w, min_periods=1).std()
                df[f'{col}_rolling_min_{w}'] = s.rolling(w, min_periods=1).min()
                df[f'{col}_rolling_max_{w}'] = s.rolling(w, min_periods=1).max()
                
        return df

    def add_weather_interactions(self, df):
        if all(col in df.columns for col in ['temperature_2m', 'relative_humidity_2m']):
            df['feels_like'] = df['temperature_2m'] + 0.3 * (df['relative_humidity_2m'] / 100 - 0.5)
        
        if all(col in df.columns for col in ['temperature_2m', 'wind_speed_10m']):
            df['wind_chill'] = 13.12 + 0.6215 * df['temperature_2m'] - 11.37 * (df['wind_speed_10m'] ** 0.16) + 0.3965 * df['temperature_2m'] * (df['wind_speed_10m'] ** 0.16)
            
        return df

    def ensure_features(self, df, features):
        self.required_features = features
        for f in features:
            if f not in df.columns:
                df[f] = 0.0
        return df

class AdvancedWeatherPredictor:
    def __init__(self):
        self.models = {}
        self.features = []
        self.engineer = AdvancedFeatureEngineer()
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained models with fallback"""
        targets = ['temperature_2m', 'precipitation', 'wind_speed_10m', 
                  'relative_humidity_2m', 'cloud_cover', 'pressure_msl']
        
        model_loaded = False
        for t in targets:
            try:
                self.models[t] = joblib.load(f"enhanced_weather_model_{t}.pkl")
                model_loaded = True
            except Exception as e:
                st.warning(f"Model for {t} not found: {e}")
                continue
                
        try:
            self.features = joblib.load("enhanced_model_features.pkl")
        except:
            # Fallback features
            self.features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                           'temperature_2m_lag_1', 'temperature_2m_lag_3',
                           'precipitation_lag_1', 'precipitation_lag_3',
                           'relative_humidity_2m_lag_1', 'wind_speed_10m_lag_1']
            
        if not model_loaded:
            st.info("Using statistical forecasting (ML models not available)")

    def predict_condition(self, row):
        """Enhanced weather condition prediction"""
        t = row.get('temperature_2m', 20)
        p = row.get('precipitation', 0)
        c = row.get('cloud_cover', 50)
        w = row.get('wind_speed_10m', 5)
        h = row.get('relative_humidity_2m', 65)
        
        # Enhanced condition logic
        if p > 8: return "Heavy Rain "
        elif p > 3: return "Rain "
        elif p > 0.5: return "Light Rain "
        elif w > 25: return "Windy "
        elif w > 15: return "Breezy "
        elif c > 90 and h > 85: return "Foggy "
        elif c > 85: return "Overcast "
        elif c > 60: return "Cloudy "
        elif c > 30: return "Partly Cloudy "
        elif t > 30: return "Hot "
        elif t < 5: return "Cold "
        else: return "Clear Sky"

    def calculate_comfort_index(self, temp, humidity, wind_speed):
        """Calculate temperature-humidity-wind comfort index"""
        # Simple comfort calculation (0-100 scale)
        temp_comfort = 100 - abs(22 - temp) * 3  # Ideal around 22°C
        humidity_comfort = 100 - abs(50 - humidity) * 1.2  # Ideal around 50%
        wind_comfort = min(100, wind_speed * 5)  # Some wind is good
        
        return (temp_comfort + humidity_comfort + wind_comfort) / 3

    def generate_forecast(self, historical_data, hours=24, start_time=None):
        """Generate weather forecast with enhanced features"""
        if start_time is None:
            start_time = datetime.now()
            
        df = historical_data.copy()
        
        # Feature engineering
        df = self.engineer.add_temporal_features(df)
        df = self.engineer.add_lagged_features(df)
        df = self.engineer.add_rolling_features(df)
        df = self.engineer.add_weather_interactions(df)
        df = self.engineer.ensure_features(df, self.features)
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        preds = []
        state = df.tail(100).copy()

        for h in range(hours):
            t = start_time + timedelta(hours=h+1)
            X = state.iloc[-1:][self.features]

            pred = {'time': t}
            
            if self.models:
                # Use ML models if available
                for target, model in self.models.items():
                    try:
                        val = float(model.predict(X)[0])
                        # Apply realistic constraints
                        if 'precipitation' in target:
                            val = max(0, val)
                        elif 'cloud' in target or 'humidity' in target:
                            val = np.clip(val, 0, 100)
                        elif 'wind' in target:
                            val = max(0, val)
                        pred[target] = val
                    except Exception as e:
                        # Fallback to persistence forecast
                        pred[target] = state[target].iloc[-1] if target in state.columns else 0
            else:
                # Statistical forecasting fallback
                pred.update(self._statistical_forecast(state, h))
            
            # Add derived metrics
            pred['weather_condition'] = self.predict_condition(pred)
            pred['comfort_index'] = self.calculate_comfort_index(
                pred.get('temperature_2m', 20),
                pred.get('relative_humidity_2m', 65),
                pred.get('wind_speed_10m', 5)
            )
            
            preds.append(pred)
            state = pd.concat([state, pd.DataFrame([pred])], ignore_index=True)

        return pd.DataFrame(preds)

    def _statistical_forecast(self, state, hour_offset):
        """Statistical forecasting when ML models are unavailable"""
        base_temp = state['temperature_2m'].iloc[-1]
        base_humidity = state['relative_humidity_2m'].iloc[-1]
        base_wind = state['wind_speed_10m'].iloc[-1]
        base_pressure = state['pressure_msl'].iloc[-1]
        
        # Add diurnal cycle and randomness
        hour = (datetime.now().hour + hour_offset + 1) % 24
        diurnal_temp = 4 * np.sin(2 * np.pi * (hour - 14) / 24)  # Peak at 2 PM
        
        return {
            'temperature_2m': base_temp + diurnal_temp + np.random.normal(0, 0.5),
            'precipitation': max(0, np.random.exponential(0.3)),
            'wind_speed_10m': max(0, base_wind + np.random.normal(0, 0.8)),
            'relative_humidity_2m': np.clip(base_humidity + np.random.normal(0, 3), 30, 95),
            'cloud_cover': np.clip(50 + 30 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 10), 0, 100),
            'pressure_msl': base_pressure + np.random.normal(0, 0.5)
        }

# ==================== ENHANCED DATA SOURCES ====================
@st.cache_data(ttl=300)
def get_current_weather():
    """Get current weather with multiple fallback options"""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': 27.7172, 'longitude': 85.3240,
            'current': 'temperature_2m,relative_humidity_2m,precipitation,surface_pressure,cloud_cover,wind_speed_10m,is_day',
            'hourly': 'temperature_2m,precipitation',
            'timezone': 'Asia/Kathmandu',
            'forecast_days': 1
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        current = data['current']
        return {
            'temperature_2m': current['temperature_2m'],
            'relative_humidity_2m': current['relative_humidity_2m'],
            'precipitation': current['precipitation'],
            'surface_pressure': current['surface_pressure'],
            'cloud_cover': current['cloud_cover'],
            'wind_speed_10m': current['wind_speed_10m'],
            'is_day': current['is_day']
        }
    except Exception as e:
        st.warning(f"Live weather data unavailable: {e}. Using sample data.")
        # Return realistic sample data for Kathmandu
        return {
            'temperature_2m': 19.2, 
            'relative_humidity_2m': 68, 
            'precipitation': 0.0,
            'surface_pressure': 868, 
            'cloud_cover': 42, 
            'wind_speed_10m': 6.8,
            'is_day': 1
        }

@st.cache_data(ttl=3600)
def load_historical_data():
    """Load historical data with enhanced synthetic data generation"""
    try:
        df = pd.read_csv("open-meteo-27.75N85.50E1293m.csv")
        df['time'] = pd.to_datetime(df['time'])
        st.success(" Historical data loaded successfully")
        return df.sort_values('time').reset_index(drop=True)
    except Exception as e:
        st.warning(f"Historical data file not found: {e}. Generating synthetic data.")
        
        # Enhanced synthetic data generation
        dates = pd.date_range("2023-01-01", periods=8784, freq="H")
        np.random.seed(42)  # For reproducibility
        
        # Base patterns for Kathmandu
        base_temp = 18 + 8 * np.sin(2 * np.pi * (dates.month - 3) / 12)  # Seasonal
        diurnal_temp = 7 * np.sin(2 * np.pi * (dates.hour - 14) / 24)    # Daily
        temp_noise = np.random.normal(0, 1.5, len(dates))
        
        return pd.DataFrame({
            'time': dates,
            'temperature_2m': base_temp + diurnal_temp + temp_noise,
            'precipitation': np.random.exponential(0.2, len(dates)),
            'wind_speed_10m': 4 + 3 * np.random.random(len(dates)),
            'relative_humidity_2m': 65 + 20 * np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 10, len(dates)),
            'pressure_msl': 870 + 5 * np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 2, len(dates)),
            'cloud_cover': 50 + 30 * np.random.random(len(dates))
        })

# ==================== UTILITY FUNCTIONS ====================
def get_weather_alerts(forecast_df):
    """Generate weather alerts based on forecast"""
    alerts = []
    
    max_temp = forecast_df['temperature_2m'].max()
    total_rain = forecast_df['precipitation'].sum()
    max_wind = forecast_df['wind_speed_10m'].max()
    
    if max_temp > 32:
        alerts.append(" High temperature warning: Stay hydrated!")
    if max_temp < 5:
        alerts.append(" Low temperature alert: Dress warmly!")
    if total_rain > 20:
        alerts.append(" Heavy rainfall expected: Potential flooding!")
    elif total_rain > 10:
        alerts.append(" Significant rainfall: Carry umbrella!")
    if max_wind > 30:
        alerts.append(" High wind warning: Secure loose objects!")
    elif max_wind > 20:
        alerts.append(" Strong winds expected!")
        
    return alerts

def create_weather_summary(forecast_df):
    """Create a human-readable weather summary"""
    avg_temp = forecast_df['temperature_2m'].mean()
    total_rain = forecast_df['precipitation'].sum()
    conditions = forecast_df['weather_condition'].value_counts()
    
    dominant_condition = conditions.index[0] if len(conditions) > 0 else "Clear Sky"
    
    summary = f"""
    The weather will be characterized by **{dominant_condition.lower()}** conditions. 
    Average temperature will be around **{avg_temp:.1f}°C** with **{total_rain:.1f} mm** of total precipitation.
    """
    
    if total_rain > 5:
        summary += " Expect wet conditions throughout the period."
    elif avg_temp > 25:
        summary += " Warm conditions expected."
    elif avg_temp < 10:
        summary += " Cool conditions expected."
        
    return summary

# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown("<div class='header-title'>Kathmandu Weather Forecasting System</div>", unsafe_allow_html=True)
    st.markdown("<div class='header-subtitle'>AI-Powered Weather Prediction & Analytics Dashboard</div>", unsafe_allow_html=True)
    
    # Initialize components
    predictor = AdvancedWeatherPredictor()
    hist_data = load_historical_data()
    
    # Store in session state for persistence
    st.session_state.historical_data = hist_data

    # Sidebar
    with st.sidebar:
        st.markdown("<h3 style='color:#3b82f6; margin-bottom: 2rem;'> Navigation</h3>", unsafe_allow_html=True)
        page = st.radio("Select Page", 
                       [" Dashboard", " Detailed Forecast", " Historical Analysis", " About"],
                       label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("###  Location Info")
        st.write("**City**: Kathmandu Valley")
        st.write("**Coordinates**: 27.7172°N, 85.3240°E")
        st.write("**Elevation**: 1,293 meters")
        st.write("**Current Time**:", datetime.now().strftime("%d %B %Y, %H:%M"))
        
        st.markdown("---")
        st.markdown("###  Settings")
        if page == " Detailed Forecast":
            hours = st.slider("Forecast Hours", 6, 168, 24, help="Select number of hours to forecast")
            st.session_state.forecast_hours = hours
        
        st.markdown("---")
        st.markdown("<div class='data-source'>Data Source: Open-Meteo API & Historical Records</div>", unsafe_allow_html=True)

    # ==================== DASHBOARD PAGE ====================
    if page == " Dashboard":
        current = get_current_weather()
        
        # Current Weather Metrics
        st.markdown("<div class='section-header'>Current Weather Conditions</div>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{current['temperature_2m']:.1f}°C</div>
                <div class='metric-label'>Temperature</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{current['relative_humidity_2m']:.0f}%</div>
                <div class='metric-label'>Humidity</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{current['wind_speed_10m']:.1f} km/h</div>
                <div class='metric-label'>Wind Speed</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{current['surface_pressure']:.0f} hPa</div>
                <div class='metric-label'>Pressure</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick Forecast
        st.markdown("<div class='section-header'>Next 24-Hour Outlook</div>", unsafe_allow_html=True)
        
        with st.spinner("Generating quick forecast..."):
            quick_forecast = predictor.generate_forecast(hist_data, hours=24)
        
        # Display weather alerts
        alerts = get_weather_alerts(quick_forecast)
        if alerts:
            for alert in alerts:
                st.markdown(f'<div class="weather-alert">{alert}</div>', unsafe_allow_html=True)
        
        # Forecast cards in rows of 6
        st.markdown("#### Hourly Forecast")
        for i in range(0, min(24, len(quick_forecast)), 6):
            cols = st.columns(6)
            for j, col in enumerate(cols):
                if i + j < len(quick_forecast):
                    row = quick_forecast.iloc[i + j]
                    with col:
                        st.markdown(f"""
                        <div class='forecast-card'>
                            <div style='font-weight:600; color:#1e40af; font-size:1.1rem;'>{row['time'].strftime('%H:%M')}</div>
                            <div style='font-size:0.9rem; color:#64748b;'>{row['time'].strftime('%a')}</div>
                            <div class='condition-badge'>{row['weather_condition']}</div>
                            <div style='margin-top:12px; font-size:0.95rem;'>
                                <div> {row['temperature_2m']:.1f}°C</div>
                                <div> {row['precipitation']:.1f}mm</div>
                                <div> {row['wind_speed_10m']:.1f} km/h</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Comfort Index Chart
        st.markdown("<div class='section-header'>Comfort Index</div>", unsafe_allow_html=True)
        fig_comfort = px.area(quick_forecast, x='time', y='comfort_index',
                             title="Temperature-Humidity-Wind Comfort Index",
                             labels={'comfort_index': 'Comfort Index (0-100)', 'time': 'Time'})
        fig_comfort.update_layout(height=400)
        st.plotly_chart(fig_comfort, use_container_width=True)

    # ==================== DETAILED FORECAST PAGE ====================
    elif page == " Detailed Forecast":
        st.markdown("<div class='section-header'>Advanced Weather Forecast</div>", unsafe_allow_html=True)
        
        # Forecast controls
        col1, col2, col3 = st.columns([2,1,1])
        with col1:
            hours = st.slider("Forecast Duration (hours)", 6, 168, 24, key="detail_forecast_hours")
        with col2:
            include_comfort = st.checkbox("Comfort Index", value=True)
        with col3:
            st.write("")  # Spacer
            if st.button(" Generate Forecast", use_container_width=True):
                with st.spinner("Generating advanced forecast..."):
                    forecast_data = predictor.generate_forecast(hist_data, hours=hours)
                    st.session_state.current_forecast = forecast_data
                    st.session_state.forecast_generated = True
                    st.success(f" Forecast generated for {hours} hours!")
        
        if st.session_state.forecast_generated and st.session_state.current_forecast is not None:
            df = st.session_state.current_forecast
            
            # Weather Summary
            st.markdown("####  Forecast Summary")
            summary = create_weather_summary(df)
            st.info(summary)
            
            # Key Metrics
            st.markdown("####  Key Forecast Metrics")
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1: st.metric("Max Temp", f"{df['temperature_2m'].max():.1f}°C")
            with m2: st.metric("Min Temp", f"{df['temperature_2m'].min():.1f}°C")
            with m3: st.metric("Total Rain", f"{df['precipitation'].sum():.1f} mm")
            with m4: st.metric("Max Wind", f"{df['wind_speed_10m'].max():.1f} km/h")
            with m5: st.metric("Avg Humidity", f"{df['relative_humidity_2m'].mean():.0f}%")
            
            # Interactive Charts
            st.markdown("####  Detailed Forecast Charts")
            
            # Create subplots based on selections
            if include_comfort:
                fig = make_subplots(rows=3, cols=2, 
                                  subplot_titles=("Temperature (°C)", "Precipitation (mm/h)", 
                                                "Wind Speed (km/h)", "Humidity (%)", 
                                                "Cloud Cover (%)", "Comfort Index"))
                
                fig.add_trace(go.Scatter(x=df['time'], y=df['temperature_2m'], name="Temp", 
                                       line=dict(width=3, color='#EF553B')), row=1, col=1)
                fig.add_trace(go.Bar(x=df['time'], y=df['precipitation'], name="Rain", 
                                   marker_color='#636EFA'), row=1, col=2)
                fig.add_trace(go.Scatter(x=df['time'], y=df['wind_speed_10m'], name="Wind", 
                                       line=dict(width=3, color='#00CC96')), row=2, col=1)
                fig.add_trace(go.Scatter(x=df['time'], y=df['relative_humidity_2m'], name="Humidity", 
                                       line=dict(width=3, color='#AB63FA')), row=2, col=2)
                fig.add_trace(go.Scatter(x=df['time'], y=df['cloud_cover'], name="Cloud", 
                                       line=dict(width=3, color='#FFA15A')), row=3, col=1)
                fig.add_trace(go.Scatter(x=df['time'], y=df['comfort_index'], name="Comfort", 
                                       line=dict(width=3, color='#19D3F3')), row=3, col=2)
            else:
                fig = make_subplots(rows=2, cols=3,
                                  subplot_titles=("Temperature (°C)", "Precipitation (mm/h)", "Wind Speed (km/h)",
                                                "Humidity (%)", "Cloud Cover (%)", "Pressure (hPa)"))
                
                fig.add_trace(go.Scatter(x=df['time'], y=df['temperature_2m'], name="Temp",
                                       line=dict(width=3, color='#EF553B')), row=1, col=1)
                fig.add_trace(go.Bar(x=df['time'], y=df['precipitation'], name="Rain",
                                   marker_color='#636EFA'), row=1, col=2)
                fig.add_trace(go.Scatter(x=df['time'], y=df['wind_speed_10m'], name="Wind",
                                       line=dict(width=3, color='#00CC96')), row=1, col=3)
                fig.add_trace(go.Scatter(x=df['time'], y=df['relative_humidity_2m'], name="Humidity",
                                       line=dict(width=3, color='#AB63FA')), row=2, col=1)
                fig.add_trace(go.Scatter(x=df['time'], y=df['cloud_cover'], name="Cloud",
                                       line=dict(width=3, color='#FFA15A')), row=2, col=2)
                fig.add_trace(go.Scatter(x=df['time'], y=df['pressure_msl'], name="Pressure",
                                       line=dict(width=3, color='#FF6692')), row=2, col=3)
            
            fig.update_layout(height=800, showlegend=True, title_text="Comprehensive Weather Forecast")
            st.plotly_chart(fig, use_container_width=True)
            
            # Data Table
            st.markdown("####  Detailed Data Table")
            display_df = df.copy()
            display_df['time'] = display_df['time'].dt.strftime("%Y-%m-%d %H:%M")
            display_df = display_df[['time', 'temperature_2m', 'precipitation', 'wind_speed_10m',
                                   'relative_humidity_2m', 'cloud_cover', 'pressure_msl', 'weather_condition']]
            display_df.columns = ['Time', 'Temp (°C)', 'Rain (mm)', 'Wind (km/h)', 
                                'Humidity (%)', 'Cloud (%)', 'Pressure (hPa)', 'Condition']
            
            st.dataframe(display_df.style.format({
                'Temp (°C)': '{:.1f}', 'Rain (mm)': '{:.2f}', 'Wind (km/h)': '{:.1f}',
                'Humidity (%)': '{:.0f}', 'Cloud (%)': '{:.0f}', 'Pressure (hPa)': '{:.1f}'
            }), use_container_width=True, height=400)
            
            # Download Section
            st.markdown("####  Download Forecast")
            csv = df.to_csv(index=False)
            st.download_button(" Download Full Forecast (CSV)", csv, 
                             f"kathmandu_forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", 
                             "text/csv")

    # ==================== HISTORICAL ANALYSIS PAGE ====================
    elif page == " Historical Analysis":
        st.markdown("<div class='section-header'>Historical Weather Analysis</div>", unsafe_allow_html=True)
        
        df = hist_data.copy()
        
        # Statistics
        st.markdown("####  Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Records", f"{len(df):,}")
        with col2: st.metric("Date Range", f"{df['time'].min().strftime('%d %b %Y')} to {df['time'].max().strftime('%d %b %Y')}")
        with col3: st.metric("Avg Temperature", f"{df['temperature_2m'].mean():.1f}°C")
        with col4: st.metric("Total Rainfall", f"{df['precipitation'].sum():.1f} mm")
        
        # Time period selection
        st.markdown("####  Historical Trends")
        col1, col2 = st.columns(2)
        with col1:
            period = st.selectbox("Analysis Period", 
                                ["Last 7 days", "Last 30 days", "Last 90 days", "Last year", "Full dataset"])
        with col2:
            variable = st.selectbox("Variable to Analyze", 
                                  ["temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m"])
        
        # Filter data based on selection
        if period == "Last 7 days":
            plot_data = df.tail(168)  # 7 days * 24 hours
        elif period == "Last 30 days":
            plot_data = df.tail(720)  # 30 days * 24 hours
        elif period == "Last 90 days":
            plot_data = df.tail(2160)  # 90 days * 24 hours
        elif period == "Last year":
            plot_data = df.tail(8760)  # 365 days * 24 hours
        else:
            plot_data = df
        
        # Create historical plot
        fig_hist = px.line(plot_data, x='time', y=variable,
                          title=f"Historical {variable.replace('_', ' ').title()} - {period}",
                          labels={variable: variable.replace('_', ' ').title(), 'time': 'Date'})
        fig_hist.update_layout(height=500)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Monthly averages
        st.markdown("####  Monthly Climate Patterns")
        monthly_data = df.copy()
        monthly_data['month'] = monthly_data['time'].dt.month
        monthly_avg = monthly_data.groupby('month').agg({
            'temperature_2m': 'mean',
            'precipitation': 'sum',
            'wind_speed_10m': 'mean'
        }).reset_index()
        
        fig_monthly = make_subplots(rows=1, cols=3, subplot_titles=("Average Temperature", "Total Precipitation", "Average Wind Speed"))
        fig_monthly.add_trace(go.Scatter(x=monthly_avg['month'], y=monthly_avg['temperature_2m'], 
                                       name="Temperature", line=dict(width=3)), row=1, col=1)
        fig_monthly.add_trace(go.Bar(x=monthly_avg['month'], y=monthly_avg['precipitation'], 
                                   name="Precipitation"), row=1, col=2)
        fig_monthly.add_trace(go.Scatter(x=monthly_avg['month'], y=monthly_avg['wind_speed_10m'], 
                                       name="Wind Speed", line=dict(width=3)), row=1, col=3)
        fig_monthly.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_monthly, use_container_width=True)

    # ==================== ABOUT PAGE ====================
    else:
        st.markdown("<div class='section-header'>About This Dashboard</div>", unsafe_allow_html=True)
        
        st.markdown("""
        ###  Kathmandu Weather Forecasting System
        
        This advanced weather dashboard provides real-time weather monitoring, AI-powered forecasting, 
        and historical analysis for the Kathmandu Valley region.
        
        ####  Features
        
        - **Real-time Monitoring**: Current weather conditions with live updates
        - **AI Forecasting**: Machine learning-powered weather predictions
        - **Historical Analysis**: Long-term weather pattern visualization
        - **Comfort Index**: Temperature-humidity-wind comfort calculations
        - **Weather Alerts**: Automated severe weather notifications
        
        ####  Data Sources
        
        - **Live Data**: Open-Meteo Weather API
        - **Historical Data**: Local weather station records
        - **Forecast Models**: Ensemble machine learning algorithms
        
        ####  Technical Implementation
        
        - **Backend**: Python with scikit-learn models
        - **Frontend**: Streamlit interactive dashboard
        - **Visualization**: Plotly charts and graphs
        - **Data Processing**: Pandas for time series analysis
        
        ####  Coverage Area
        
        - **City**: Kathmandu Metropolitan City
        - **Coordinates**: 27.7172°N, 85.3240°E  
        - **Elevation**: 1,293 meters above sea level
        - **Climate**: Subtropical highland climate
        
        For questions or issues, please contact the development team.
        """)
        
        st.markdown("---")
        st.markdown("<div class='data-source'>Developed with ❤️ for Kathmandu Valley</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()