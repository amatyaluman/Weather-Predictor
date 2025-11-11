# weather_dashboard.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import holidays
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Weather Intelligence System",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    .main-header {
        font-size: 2rem;
        color: #6873de;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border: 1px solid #e3f2fd;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: #1a237e;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.3rem;
        box-shadow: 0 3px 8px rgba(0,0,0,0.2);
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-label {
        font-size: 1.1em;
        margin-bottom: 0.5em;
        opacity: 0.9;
    }
    .metric-value {
        font-size: 1.8em;
        font-weight: bold;
        margin-bottom: 0.2em;
    }
    .metric-caption {
        font-size: 0.8em;
        opacity: 0.8;
        margin-top: 0.5em;
    }
    .prediction-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.3rem;
        border-left: 4px solid #1a237e;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .section-header {
        font-size: 1.4rem;
        color: #6873de;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #5c6bc0;
        font-weight: 600;
    }
    .icon {
        margin-right: 8px;
        color: #1a237e;
    }
    .metric-icon {
        font-size: 1.2em;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedFeatureEngineer:
    def add_temporal_features(self, df):
        df['hour'] = df['time'].dt.hour
        df['month'] = df['time'].dt.month
        df['day_of_year'] = df['time'].dt.dayofyear
        df['weekday'] = df['time'].dt.weekday
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        return df
    
    def add_meteorological_features(self, df):
        df['diurnal_phase'] = 6 * np.sin(2 * np.pi * (df['hour'] - 14) / 24)
        df['seasonal_adj'] = df['month'].apply(self._get_seasonal_adjustment)
        df['monsoon_effect'] = df['month'].apply(lambda x: 0.4 if x in [6,7,8,9] else 0.1)
        if 'pressure_msl' in df.columns:
            for window in [3, 6, 12, 24]:
                df[f'pressure_rolling_mean_{window}'] = df['pressure_msl'].shift(1).rolling(window=window, min_periods=1).mean()
        return df
    
    def _get_seasonal_adjustment(self, month):
        adjustments = {1:-3, 2:-1, 3:2, 4:4, 5:5, 6:3, 7:1, 8:1, 9:2, 10:1, 11:-1, 12:-2}
        return adjustments.get(month, 0)
    
    def add_lagged_features(self, df, lags=[1, 3, 6, 12, 24]):
        base_columns = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 
                       'precipitation', 'pressure_msl', 'cloud_cover']
        available_columns = [col for col in base_columns if col in df.columns]
        for col in available_columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        return df
    
    def add_rolling_features(self, df, windows=[3, 6, 12, 24]):
        base_columns = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 
                       'pressure_msl', 'cloud_cover']
        available_columns = [col for col in base_columns if col in df.columns]
        for col in available_columns:
            for window in windows:
                shifted = df[col].shift(1)
                df[f'{col}_rolling_mean_{window}'] = shifted.rolling(window=window, min_periods=1).mean()
                df[f'{col}_rolling_std_{window}'] = shifted.rolling(window=window, min_periods=1).std()
                df[f'{col}_rolling_min_{window}'] = shifted.rolling(window=window, min_periods=1).min()
                df[f'{col}_rolling_max_{window}'] = shifted.rolling(window=window, min_periods=1).max()
        return df

class AtmosphericPatternLearner:
    def __init__(self):
        self.feature_engineer = AdvancedFeatureEngineer()
        self.patterns_learned = False
        self.monthly_patterns = {}
        self.diurnal_patterns = {}
        self.weather_transitions = {}
        
    def learn_patterns_from_data(self, historical_data):
        if historical_data is None or len(historical_data) == 0:
            st.warning("No historical data available for pattern learning")
            return False
        try:
            df = historical_data.copy()
            df = self.feature_engineer.add_temporal_features(df)
            df = self.feature_engineer.add_meteorological_features(df)
            self._learn_monthly_patterns(df)
            self._learn_diurnal_patterns(df)
            self._learn_weather_transitions(df)
            self.patterns_learned = True
            return True
        except Exception as e:
            st.warning(f"Pattern learning limited: {e}")
            return False
    
    def _learn_monthly_patterns(self, df):
        monthly_stats = df.groupby('month').agg({
            'temperature_2m': ['mean', 'std'],
            'precipitation': ['mean', 'std', lambda x: (x > 0.1).mean()],
            'wind_speed_10m': ['mean', 'std'],
            'relative_humidity_2m': ['mean', 'std'],
            'cloud_cover': ['mean', 'std']
        }).round(3)
        self.monthly_patterns = monthly_stats.to_dict()
    
    def _learn_diurnal_patterns(self, df):
        hourly_stats = df.groupby('hour').agg({
            'temperature_2m': ['mean', 'std'],
            'precipitation': ['mean', 'std'],
            'wind_speed_10m': ['mean', 'std'],
            'relative_humidity_2m': ['mean', 'std']
        }).round(3)
        self.diurnal_patterns = hourly_stats.to_dict()
    
    def _learn_weather_transitions(self, df):
        df['weather_state'] = df.apply(self._classify_weather_state, axis=1)
        transitions = []
        for i in range(1, len(df)):
            if df['time'].iloc[i] - df['time'].iloc[i-1] == timedelta(hours=1):
                transitions.append((df['weather_state'].iloc[i-1], df['weather_state'].iloc[i]))
        transition_counts = {}
        for prev, curr in transitions:
            if prev not in transition_counts:
                transition_counts[prev] = {}
            transition_counts[prev][curr] = transition_counts[prev].get(curr, 0) + 1
        self.weather_transitions = {}
        for prev_state, next_states in transition_counts.items():
            total = sum(next_states.values())
            self.weather_transitions[prev_state] = {k: v/total for k, v in next_states.items()}
    
    def _classify_weather_state(self, row):
        temp = row.get('temperature_2m', 20)
        precip = row.get('precipitation', 0)
        cloud = row.get('cloud_cover', 50)
        if precip > 5: return 'rain_heavy'
        elif precip > 1: return 'rain_light'
        elif cloud > 80: return 'cloudy'
        elif cloud > 40: return 'partly_cloudy'
        elif temp > 28: return 'hot'
        elif temp > 20: return 'warm'
        elif temp > 10: return 'cool'
        else: return 'cold'

class RealisticWeatherPredictor:
    def __init__(self):
        self.models = None
        self.features = None
        self.engineer = AdvancedFeatureEngineer()
        self.pattern_learner = AtmosphericPatternLearner()
        try:
            self.np_holidays = holidays.Nepal()
        except:
            self.np_holidays = {}

    def load_models(self):
        try:
            self.models = joblib.load("weather_models.pkl")
            self.features = joblib.load("model_features.pkl")
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False

    def predict_weather_condition(self, row):
        try:
            temp = row.get('temperature_2m', 20)
            precip = row.get('precipitation', 0)
            cloud = row.get('cloud_cover', 0)
            wind = row.get('wind_speed_10m', 0)
            hum = row.get('relative_humidity_2m', 60)

            if precip > 8 and cloud > 85 and wind > 25:
                return 'Thunderstorm', 'fas fa-bolt'
            elif precip > 5:
                return 'Heavy Rain', 'fas fa-cloud-showers-heavy'
            elif precip > 2:
                return 'Moderate Rain', 'fas fa-cloud-rain'
            elif precip > 0.5:
                return 'Light Rain', 'fas fa-cloud-drizzle'
            elif precip > 0.1:
                return 'Drizzle', 'fas fa-cloud-drizzle'
            elif temp < 0 and precip > 0.1:
                return 'Snow', 'fas fa-snowflake'
            elif hum > 90 and cloud > 80 and wind < 5:
                return 'Fog', 'fas fa-smog'
            elif cloud > 80:
                return 'Overcast', 'fas fa-cloud'
            elif cloud > 60:
                return 'Mostly Cloudy', 'fas fa-cloud-sun'
            elif cloud > 30:
                return 'Partly Cloudy', 'fas fa-cloud-sun'
            elif wind > 30:
                return 'Windy', 'fas fa-wind'
            elif wind > 20:
                return 'Breezy', 'fas fa-wind'
            else:
                if temp > 30:
                    return 'Hot', 'fas fa-temperature-high'
                elif temp > 25:
                    return 'Warm', 'fas fa-sun'
                elif temp > 18:
                    return 'Pleasant', 'fas fa-sun'
                elif temp > 10:
                    return 'Cool', 'fas fa-temperature-low'
                else:
                    return 'Cold', 'fas fa-temperature-low'
        except:
            return 'Unknown', 'fas fa-question'

    def generate_realistic_predictions(self, historical_data, hours=24, start_date=None):
        try:
            if not self.pattern_learner.patterns_learned:
                self.pattern_learner.learn_patterns_from_data(historical_data)
            
            if self.models:
                hist = historical_data.copy()
                if hist is None or len(hist) == 0:
                    raise ValueError("No historical data")
                
                now = start_date if start_date else datetime.now()
                hist = hist[hist['time'] < now]
                
                cur = get_current_weather()
                if cur:
                    current_row = {
                        'time': now,
                        'temperature_2m': cur.get('temperature_2m', 20),
                        'relative_humidity_2m': cur.get('relative_humidity_2m', 60),
                        'precipitation': cur.get('precipitation', 0),
                        'pressure_msl': cur.get('surface_pressure', 870),
                        'cloud_cover': cur.get('cloud_cover', 50),
                        'wind_speed_10m': cur.get('wind_speed_10m', 5),
                    }
                    hist = pd.concat([hist, pd.DataFrame([current_row])], ignore_index=True)
                
                hist = self.engineer.add_temporal_features(hist)
                hist = self.engineer.add_meteorological_features(hist)
                hist = self.engineer.add_lagged_features(hist)
                hist = self.engineer.add_rolling_features(hist)
                hist['is_holiday'] = hist['time'].apply(lambda x: int(x in self.np_holidays))
                hist = hist.fillna(method='ffill').fillna(0)
                
                lags = [1, 3, 6, 12, 24]
                windows = [3, 6, 12, 24]
                max_shift = max(max(lags), max(windows))
                tail = hist.tail(max_shift).copy()
                
                preds = []

                for h in range(hours):
                    future_time = now + timedelta(hours=h + 1)

                    # Build full feature row
                    new_row = pd.DataFrame({'time': [future_time]})
                    temp_df = pd.concat([tail, new_row], ignore_index=True)

                    temp_df = self.engineer.add_temporal_features(temp_df)
                    temp_df = self.engineer.add_meteorological_features(temp_df)
                    temp_df = self.engineer.add_lagged_features(temp_df)
                    temp_df = self.engineer.add_rolling_features(temp_df)
                    temp_df['is_holiday'] = int(future_time in self.np_holidays)
                    temp_df = temp_df.fillna(method='ffill').fillna(0)

                    # Extract feature vector (1 row)
                    X_row = temp_df.iloc[-1:][self.features]

                    row = {'time': future_time}
                    for target, model in self.models.items():
                        pred = model.predict(X_row.values)[0]
                        row[target] = pred
                        temp_df.at[temp_df.index[-1], target] = pred

                    cond, icon = self.predict_weather_condition(row)
                    row['weather_condition'] = cond
                    row['condition_icon'] = icon

                    preds.append(row)

                    # Update rolling window
                    tail = pd.concat([tail.iloc[1:], temp_df.iloc[-1:]], ignore_index=True)
                
                return pd.DataFrame(preds)
            
            else:
                # Fallback pattern-based
                preds = []
                now = start_date if start_date else datetime.now()
                cur = get_current_weather()
                if cur:
                    current_temp = cur.get('temperature_2m', 20)
                    current_humidity = cur.get('relative_humidity_2m', 60)
                    current_pressure = cur.get('surface_pressure', 870)
                    current_wind = cur.get('wind_speed_10m', 5)
                    current_cloud = cur.get('cloud_cover', 50)
                else:
                    current_temp = 18
                    current_humidity = 65
                    current_pressure = 870
                    current_wind = 8
                    current_cloud = 40

                for h in range(hours):
                    prediction_time = now + timedelta(hours=h)
                    hour_of_day = prediction_time.hour
                    month = prediction_time.month
                    
                    temp_pred = self._predict_temperature(current_temp, hour_of_day, month, h)
                    precip_pred = self._predict_precipitation(hour_of_day, month, h)
                    wind_pred = self._predict_wind_speed(current_wind, hour_of_day, h)
                    humidity_pred = self._predict_humidity(current_humidity, temp_pred, precip_pred, h)
                    cloud_pred = self._predict_cloud_cover(current_cloud, precip_pred, humidity_pred, h)
                    pressure_pred = self._predict_pressure(current_pressure, h)

                    row = {
                        'time': prediction_time,
                        'temperature_2m': round(temp_pred, 1),
                        'precipitation': round(precip_pred, 1),
                        'wind_speed_10m': round(wind_pred, 1),
                        'relative_humidity_2m': round(humidity_pred),
                        'cloud_cover': round(cloud_pred),
                        'pressure_msl': round(pressure_pred, 1)
                    }

                    cond, icon = self.predict_weather_condition(row)
                    row['weather_condition'] = cond
                    row['condition_icon'] = icon

                    preds.append(row)

                return pd.DataFrame(preds)

        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None

    def _predict_temperature(self, current_temp, hour, month, hour_ahead):
        diurnal_variation = 6 * np.sin(2 * np.pi * (hour - 14) / 24)
        seasonal_adj = self.pattern_learner.feature_engineer._get_seasonal_adjustment(month)
        autocorr = 0.8 ** (hour_ahead / 6)
        random_variation = np.random.normal(0, 1.5)
        predicted_temp = current_temp * autocorr + diurnal_variation + seasonal_adj + random_variation
        return max(-10, min(40, predicted_temp))

    def _predict_precipitation(self, hour, month, hour_ahead):
        monsoon_months = [6, 7, 8, 9]
        base_prob = 0.3 if month in monsoon_months else 0.05
        diurnal_effect = 0.2 * np.sin(2 * np.pi * (hour - 16) / 24)
        horizon_effect = 0.8 ** (hour_ahead / 12)
        rain_probability = base_prob + diurnal_effect * horizon_effect
        if np.random.random() < rain_probability:
            precipitation = np.random.exponential(2.0 if month in monsoon_months else 0.5)
            return min(20, precipitation)
        else:
            return 0.0

    def _predict_wind_speed(self, current_wind, hour, hour_ahead):
        diurnal_pattern = 2 * np.sin(2 * np.pi * hour / 24)
        persistence = 0.7 ** (hour_ahead / 8)
        random_variation = np.random.normal(0, 1.2)
        predicted_wind = current_wind * persistence + diurnal_pattern + random_variation
        return max(0, min(60, predicted_wind))

    def _predict_humidity(self, current_humidity, temperature, precipitation, hour_ahead):
        temp_effect = -0.5 * (temperature - 20)
        precip_effect = 20 if precipitation > 0.1 else 0
        persistence = 0.6 ** (hour_ahead / 6)
        diurnal_effect = 10 * np.sin(2 * np.pi * (hour_ahead % 24 - 4) / 24)
        predicted_humidity = current_humidity * persistence + temp_effect + precip_effect + diurnal_effect
        return max(20, min(100, predicted_humidity))

    def _predict_cloud_cover(self, current_cloud, precipitation, humidity, hour_ahead):
        precip_effect = 40 if precipitation > 0.1 else 0
        humidity_effect = 0.3 * (humidity - 50)
        persistence = 0.5 ** (hour_ahead / 8)
        random_variation = np.random.normal(0, 10)
        predicted_cloud = current_cloud * persistence + precip_effect + humidity_effect + random_variation
        return max(0, min(100, predicted_cloud))

    def _predict_pressure(self, current_pressure, hour_ahead):
        persistence = 0.9 ** (hour_ahead / 24)
        random_variation = np.random.normal(0, 2)
        predicted_pressure = current_pressure * persistence + random_variation
        return max(860, min(890, predicted_pressure))

def get_current_weather():
    try:
        params = {
            'latitude': 27.7172,
            'longitude': 85.3240,
            'current': ['temperature_2m','relative_humidity_2m','dew_point_2m','precipitation','surface_pressure','cloud_cover','wind_speed_10m','wind_direction_10m','weather_code'],
            'timezone': 'Asia/Kathmandu'
        }
        r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10)
        d = r.json()
        return d.get('current')
    except Exception as e:
        st.warning(f"Using fallback current weather data: {e}")
        return {
            'temperature_2m':18.5,
            'relative_humidity_2m':65,
            'wind_speed_10m':8.2,
            'surface_pressure':870,
            'cloud_cover':45,
            'precipitation':0.0
        }

@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv("open-meteo-27.75N85.50E1293m.csv")
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')
        return df
    except Exception as e:
        st.warning(f"Using generated historical data: {e}")
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='H')
        np.random.seed(42)
        base_temp = 15
        seasonal_variation = 8 * np.sin(2 * np.pi * (dates.month - 3) / 12)
        diurnal_variation = 8 * np.sin(2 * np.pi * (dates.hour - 14) / 24)
        temperatures = base_temp + seasonal_variation + diurnal_variation + np.random.normal(0, 2, len(dates))
        monsoon_months = [6, 7, 8, 9]
        precip_base = np.where(np.isin(dates.month, monsoon_months), 0.3, 0.05)
        precipitation = np.random.exponential(precip_base, len(dates))
        d = {
            'time': dates,
            'temperature_2m': temperatures,
            'precipitation': precipitation,
            'wind_speed_10m': 5 + 3 * np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 1.5, len(dates)),
            'relative_humidity_2m': 60 + 20 * np.sin(2 * np.pi * (dates.hour - 4) / 24) + np.random.normal(0, 5, len(dates)),
            'pressure_msl': 870 + 5 * np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 1, len(dates)),
            'cloud_cover': 40 + 30 * np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 10, len(dates))
        }
        return pd.DataFrame(d)

def display_current_weather():
    st.markdown("## Current Weather")
    cur = get_current_weather()
    if cur:
        cols = st.columns(4)
        with cols[0]:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Temperature</div>
                <div class='metric-value'>{cur['temperature_2m']:.1f}°C</div>
            </div>
            """, unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Humidity</div>
                <div class='metric-value'>{cur['relative_humidity_2m']:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Wind Speed</div>
                <div class='metric-value'>{cur['wind_speed_10m']:.1f} km/h</div>
            </div>
            """, unsafe_allow_html=True)
        with cols[3]:
            p = cur.get('surface_pressure', 870)
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Pressure</div>
                <div class='metric-value'>{p:.0f} hPa</div>
                <div class='metric-caption'>Surface pressure at 1293m</div>
            </div>
            """, unsafe_allow_html=True)

def show_hourly_predictions():
    st.markdown("## Hourly Weather Forecast")
    st.markdown("<div class='section-header'>Forecast Settings</div>", unsafe_allow_html=True)

    predictor = RealisticWeatherPredictor()
    models_loaded = predictor.load_models()
    if models_loaded:
        st.success("ML models loaded successfully")
    else:
        st.warning("Using pattern-based weather prediction (ML models not found)")
    
    hist = load_historical_data()

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        hrs = st.selectbox("Forecast Hours", [12,24,36,48], index=1)
    with c2:
        start_date = st.date_input("Start Date", value=datetime.now())
    with c3:
        go_btn = st.button("Generate Forecast", use_container_width=True)

    if not go_btn:
        display_sample_layout()
        return

    with st.spinner("Generating realistic weather forecast..."):
        start_datetime = datetime.combine(start_date, datetime.now().time())
        pred = predictor.generate_realistic_predictions(hist, hrs, start_datetime)
        if pred is not None:
            display_prediction_dashboard(pred)

def display_sample_layout():
    st.info("Click 'Generate Forecast' to see detailed predictions")
    predictor = RealisticWeatherPredictor()
    hist = load_historical_data()
    patterns_learned = predictor.pattern_learner.learn_patterns_from_data(hist)
    if patterns_learned:
        st.success("Atmospheric patterns learned from historical data")
    else:
        st.warning("Using baseline prediction patterns")
    if hasattr(predictor.pattern_learner, 'monthly_patterns'):
        months = list(range(1, 13))
        avg_temps = [predictor.pattern_learner.monthly_patterns[('temperature_2m', 'mean')].get(m, 18) for m in months]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=avg_temps, name='Average Temperature', line=dict(width=3)))
        fig.update_layout(title="Learned Monthly Temperature Pattern", xaxis_title="Month", yaxis_title="Temperature (°C)", height=400, margin=dict(t=60))
        st.plotly_chart(fig, use_container_width=True)

def display_prediction_dashboard(pred):
    st.markdown("<div class='section-header'>Forecast Summary</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    data = [
        ("High/Low", f"{pred['temperature_2m'].max():.1f}°/{pred['temperature_2m'].min():.1f}°"),
        ("Total Rain", f"{pred['precipitation'].sum():.1f} mm"),
        ("Max Wind", f"{pred['wind_speed_10m'].max():.1f} km/h"),
        ("Rain Hours", f"{len(pred[pred['precipitation']>0.1])}")
    ]
    for col, (lbl,val) in zip(cols, data):
        with col:
            st.metric(lbl, val)

    st.markdown("<div class='section-header'>Interactive Forecast Charts</div>", unsafe_allow_html=True)
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Temperature Trend','Precipitation','Wind Speed','Humidity'), vertical_spacing=0.15)
    fig.add_trace(go.Scatter(x=pred['time'], y=pred['temperature_2m'], name='Temperature', line=dict(width=3)), row=1, col=1)
    fig.add_trace(go.Bar(x=pred['time'], y=pred['precipitation'], name='Precipitation'), row=1, col=2)
    fig.add_trace(go.Scatter(x=pred['time'], y=pred['wind_speed_10m'], name='Wind Speed', line=dict(width=3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=pred['time'], y=pred['relative_humidity_2m'], name='Humidity', line=dict(width=3)), row=2, col=2)
    fig.update_layout(height=600, showlegend=False, margin=dict(t=60))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'>Detailed Hourly Forecast</div>", unsafe_allow_html=True)
    d = pred.copy()
    d['Time'] = d['time'].dt.strftime('%Y-%m-%d %H:%M')
    d['Condition'] = d['weather_condition']
    d['Temp'] = d['temperature_2m']
    d['Rain'] = d['precipitation']
    d['Wind'] = d['wind_speed_10m']
    st.dataframe(d[['Time','Condition','Temp','Rain','Wind']].head(12), use_container_width=True, hide_index=True)

def show_historical_analysis():
    st.markdown("## Historical Weather Analysis")
    st.markdown("<div class='section-header'>Data Selection</div>", unsafe_allow_html=True)
    try:
        hist_data = load_historical_data()
        if hist_data is not None and len(hist_data) > 0:
            st.success(f"Loaded {len(hist_data)} historical records")
            col1, col2 = st.columns(2)
            with col1:
                min_date = hist_data['time'].min().date()
                max_date = hist_data['time'].max().date()
                start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
            with col2:
                end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
            mask = (hist_data['time'].dt.date >= start_date) & (hist_data['time'].dt.date <= end_date)
            filtered_data = hist_data[mask]
            if len(filtered_data) > 0:
                st.info(f"Showing {len(filtered_data)} records from {start_date} to {end_date}")
                predictor = RealisticWeatherPredictor()
                patterns_learned = predictor.pattern_learner.learn_patterns_from_data(filtered_data)
                if patterns_learned:
                    st.success("Patterns analyzed from selected historical data")
                st.markdown("<div class='section-header'>Summary Statistics</div>", unsafe_allow_html=True)
                cols = st.columns(4)
                with cols[0]:
                    avg_temp = filtered_data['temperature_2m'].mean()
                    st.metric("Average Temperature", f"{avg_temp:.1f}°C")
                with cols[1]:
                    total_rain = filtered_data['precipitation'].sum()
                    st.metric("Total Precipitation", f"{total_rain:.1f} mm")
                with cols[2]:
                    max_wind = filtered_data['wind_speed_10m'].max()
                    st.metric("Max Wind Speed", f"{max_wind:.1f} km/h")
                with cols[3]:
                    avg_humidity = filtered_data['relative_humidity_2m'].mean()
                    st.metric("Average Humidity", f"{avg_humidity:.1f}%")
                st.markdown("<div class='section-header'>Historical Trends</div>", unsafe_allow_html=True)
                fig_temp = px.line(filtered_data, x='time', y='temperature_2m', title='Temperature Trend Over Time')
                fig_temp.update_layout(height=400)
                st.plotly_chart(fig_temp, use_container_width=True)
                col1, col2 = st.columns(2)
                with col1:
                    fig_precip = px.bar(filtered_data, x='time', y='precipitation', title='Precipitation Over Time')
                    fig_precip.update_layout(height=300)
                    st.plotly_chart(fig_precip, use_container_width=True)
                with col2:
                    fig_wind = px.line(filtered_data, x='time', y='wind_speed_10m', title='Wind Speed Over Time')
                    fig_wind.update_layout(height=300)
                    st.plotly_chart(fig_wind, use_container_width=True)
                st.markdown("<div class='section-header'>Monthly Analysis</div>", unsafe_allow_html=True)
                monthly_data = filtered_data.copy()
                monthly_data['month'] = monthly_data['time'].dt.month
                monthly_avg = monthly_data.groupby('month').agg({
                    'temperature_2m': 'mean',
                    'precipitation': 'sum',
                    'wind_speed_10m': 'mean'
                }).reset_index()
                col1, col2 = st.columns(2)
                with col1:
                    fig_month_temp = px.bar(monthly_avg, x='month', y='temperature_2m', title='Average Temperature by Month')
                    st.plotly_chart(fig_month_temp, use_container_width=True)
                with col2:
                    fig_month_rain = px.bar(monthly_avg, x='month', y='precipitation', title='Total Precipitation by Month')
                    st.plotly_chart(fig_month_rain, use_container_width=True)
                st.markdown("<div class='section-header'>Historical Data</div>", unsafe_allow_html=True)
                display_data = filtered_data.copy()
                display_data['Date'] = display_data['time'].dt.strftime('%Y-%m-%d %H:%M')
                display_cols = ['Date', 'temperature_2m', 'precipitation', 'wind_speed_10m', 'relative_humidity_2m']
                display_cols = [col for col in display_cols if col in display_data.columns]
                st.dataframe(display_data[display_cols].rename(columns={
                    'temperature_2m': 'Temperature (°C)',
                    'precipitation': 'Precipitation (mm)',
                    'wind_speed_10m': 'Wind Speed (km/h)',
                    'relative_humidity_2m': 'Humidity (%)'
                }), use_container_width=True, height=300)
            else:
                st.warning("No data available for the selected date range.")
        else:
            st.warning("No historical data available. Using sample data for demonstration.")
            display_sample_historical_analysis()
    except Exception as e:
        st.error(f"Error loading historical data: {e}")
        st.info("Displaying sample historical analysis")
        display_sample_historical_analysis()

def display_sample_historical_analysis():
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    sample_data = pd.DataFrame({
        'time': dates,
        'temperature_2m': 15 + 10*np.sin(2*np.pi*dates.dayofyear/365) + np.random.normal(0, 3, len(dates)),
        'precipitation': np.random.exponential(0.5, len(dates)),
        'wind_speed_10m': 5 + 3*np.random.random(len(dates)),
        'relative_humidity_2m': 60 + 20*np.random.random(len(dates))
    })
    st.info("Sample Historical Analysis (using generated data)")
    st.markdown("<div class='section-header'>Summary Statistics</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    with cols[0]:
        avg_temp = sample_data['temperature_2m'].mean()
        st.metric("Average Temperature", f"{avg_temp:.1f}°C")
    with cols[1]:
        total_rain = sample_data['precipitation'].sum()
        st.metric("Total Precipitation", f"{total_rain:.1f} mm")
    with cols[2]:
        max_wind = sample_data['wind_speed_10m'].max()
        st.metric("Max Wind Speed", f"{max_wind:.1f} km/h")
    with cols[3]:
        avg_humidity = sample_data['relative_humidity_2m'].mean()
        st.metric("Average Humidity", f"{avg_humidity:.1f}%")
    fig_temp = px.line(sample_data, x='time', y='temperature_2m', title='Sample Temperature Trend')
    st.plotly_chart(fig_temp, use_container_width=True)

def main():
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Choose a page:", 
                       ["Dashboard", 
                        "Hourly Forecast", 
                        "Historical Analysis"],
                       index=0)

    if page == "Dashboard":
        display_current_weather()
        st.container()
        st.markdown("### Welcome to Weather Prediction System")
        st.write("A weather prediction system for Kathmandu, Nepal using machine learning models trained on historical weather data.")
        c1, c2 = st.columns(2)
        with c1:
            st.info("""
            **Features**
            - Real-time weather monitoring
            - Pattern-based accurate forecasts
            - Weather condition prediction
            - Interactive visualizations
            """)
        with c2:
            st.info("""
            **Capabilities**
            - 12-48 hour forecasts
            - Temperature trends
            - Precipitation prediction
            - Wind speed analysis
            """)

    elif page == "Hourly Forecast":
        show_hourly_predictions()

    elif page == "Historical Analysis":
        show_historical_analysis()

if __name__ == "__main__":
    main()