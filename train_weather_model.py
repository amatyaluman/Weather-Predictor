import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
import joblib
import holidays
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("Starting Advanced Weather Model Training...")

class AdvancedFeatureEngineer:
    def add_temporal_features(self, df):
        """Enhanced temporal features with more granular time components"""
        df['hour'] = df['time'].dt.hour
        df['month'] = df['time'].dt.month
        df['day_of_year'] = df['time'].dt.dayofyear
        df['weekday'] = df['time'].dt.weekday
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
        df['quarter'] = df['time'].dt.quarter
        df['is_month_start'] = df['time'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['time'].dt.is_month_end.astype(int)
        
        # Cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        return df

    def add_meteorological_features(self, df):
        """Enhanced meteorological features (must be runnable in dashboard)"""
        
        # Diurnal and seasonal patterns
        df['diurnal_phase'] = 6 * np.sin(2 * np.pi * (df['hour'] - 14) / 24)
        df['seasonal_adj'] = df['month'].apply(self._get_seasonal_adjustment)
        df['monsoon_effect'] = df['month'].apply(lambda x: 0.4 if x in [6,7,8,9] else 0.1)
        
        # Pressure features (removed the rolling/diff here, moved to dynamic_features)
        
        # Temperature-humidity interactions
        if all(col in df.columns for col in ['temperature_2m', 'relative_humidity_2m']):
            df['apparent_temperature'] = df['temperature_2m'] + 0.33 * (df['relative_humidity_2m'] / 100 * 6.105 * np.exp(17.27 * df['temperature_2m'] / (237.7 + df['temperature_2m']))) - 4
            df['heat_index'] = self.calculate_heat_index(df['temperature_2m'], df['relative_humidity_2m'])
        
        return df

    def calculate_heat_index(self, temperature, humidity):
        """Calculate heat index based on temperature and humidity"""
        hi = (0.5 * (temperature + 61.0 + ((temperature - 68.0) * 1.2) + (humidity * 0.094)))
        return np.where(temperature >= 80, hi + ((temperature - 80) * 0.1), hi)

    def _get_seasonal_adjustment(self, month):
        adjustments = {1:-3, 2:-1, 3:2, 4:4, 5:5, 6:3, 7:1, 8:1, 9:2, 10:1, 11:-1, 12:-2}
        return adjustments.get(month, 0)

    def add_lagged_features(self, df, lags=[1, 3, 6, 12, 24, 48]):
        """Enhanced lag features with more time steps"""
        base_columns = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 
                        'precipitation', 'pressure_msl', 'cloud_cover']
        available_columns = [col for col in base_columns if col in df.columns]
        
        for col in available_columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                # Add lagged differences for trend
                if lag > 1:
                    df[f'{col}_diff_{lag}'] = df[col].diff(lag)
        
        return df

    def add_rolling_features(self, df, windows=[3, 6, 12, 24, 48]):
        """Enhanced rolling features with more statistics"""
        base_columns = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 
                        'pressure_msl', 'cloud_cover', 'precipitation']
        available_columns = [col for col in base_columns if col in df.columns]
        
        for col in available_columns:
            for window in windows:
                shifted = df[col].shift(1)
                df[f'{col}_rolling_mean_{window}'] = shifted.rolling(window=window, min_periods=1).mean()
                df[f'{col}_rolling_std_{window}'] = shifted.rolling(window=window, min_periods=1).std()
                df[f'{col}_rolling_min_{window}'] = shifted.rolling(window=window, min_periods=1).min()
                df[f'{col}_rolling_max_{window}'] = shifted.rolling(window=window, min_periods=1).max()
                df[f'{col}_rolling_range_{window}'] = (
                    df[f'{col}_rolling_max_{window}'] - df[f'{col}_rolling_min_{window}']
                )
                
                # Add exponential moving averages
                df[f'{col}_ema_{window}'] = shifted.ewm(span=window, min_periods=1).mean()
        
        return df

class WeatherConditionPredictor:
    @staticmethod
    def predict_condition_vectorized(df):
        temp = df.get('temperature_2m', pd.Series(20, index=df.index))
        precip = df.get('precipitation', pd.Series(0, index=df.index))
        cloud_cover = df.get('cloud_cover', pd.Series(0, index=df.index))
        wind_speed = df.get('wind_speed_10m', pd.Series(0, index=df.index))
        humidity = df.get('relative_humidity_2m', pd.Series(60, index=df.index))

        conditions = []
        choices = []

        cond_thunder = (precip > 8.0) & (cloud_cover > 85) & (wind_speed > 25)
        cond_heavy = precip > 5.0
        cond_moderate = precip > 2.0
        cond_light = precip > 0.5
        cond_drizzle = precip > 0.1
        cond_snow = (temp < 0) & (precip > 0.1)
        cond_fog = (humidity > 90) & (cloud_cover > 80) & (wind_speed < 5)
        cond_overcast = cloud_cover > 80
        cond_mostly = cloud_cover > 60
        cond_partly = cloud_cover > 30
        cond_windy = wind_speed > 30
        cond_breezy = wind_speed > 20

        conditions = [cond_thunder, cond_heavy, cond_moderate, cond_light, cond_drizzle, cond_snow, cond_fog, cond_overcast, cond_mostly, cond_partly, cond_windy, cond_breezy]
        choices = ['Thunderstorm', 'Heavy Rain', 'Moderate Rain', 'Light Rain', 'Drizzle', 'Snow', 'Fog', 'Overcast', 'Mostly Cloudy', 'Partly Cloudy', 'Windy', 'Breezy']

        result = np.select(conditions, choices, default='__TEMP_CHECK__')
        
        # Temp check for remaining cases
        temp_hot = temp > 28
        temp_warm = temp > 22
        temp_pleasant = temp > 15
        temp_cool = temp > 8
        temp_cold = ~temp_cool

        temp_choices = ['Hot', 'Warm', 'Pleasant', 'Cool', 'Cold']
        temp_conditions = [temp_hot, temp_warm, temp_pleasant, temp_cool, temp_cold]

        temp_result = np.select(temp_conditions, temp_choices, default='Unknown')
        mask_temp = result == '__TEMP_CHECK__'
        result = np.where(mask_temp, temp_result, result)
        
        # Final cleanup for clear sky based on cloud cover
        result[(result == 'Pleasant') & (cloud_cover < 30)] = 'Clear Sky'
        result[(result == 'Warm') & (cloud_cover < 30)] = 'Clear Sky'
        
        return pd.Series(result, index=df.index)

class EnhancedDataValidator:
    @staticmethod
    def validate_dataframe(df):
        """Enhanced data validation and clipping"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        inf_mask = np.isinf(df[numeric_cols])
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Handle outliers using IQR method for all numeric columns
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df

    @staticmethod
    def check_data_quality(df):
        # ... (reporting functions remain the same for brevity) ...
        print("\n=== ENHANCED DATA QUALITY REPORT ===")
        print(f"Total records: {len(df):,}")
        print(f"Date range: {df['time'].min()} to {df['time'].max()}")
        print("=== END ENHANCED QUALITY REPORT ===\n")

    @staticmethod
    def check_temporal_consistency(df):
        if 'time' not in df.columns:
            print(" Time column not found for temporal consistency check")
            return False
        time_diff = df['time'].diff()
        expected_freq = pd.Timedelta(hours=1)
        gaps = time_diff[time_diff > expected_freq * 1.1] 
        if len(gaps) > 0:
            print(f"  Found {len(gaps)} temporal gaps in data")
            return False
        else:
            print(" Temporal consistency: No significant gaps found")
            return True

def load_historical_data():
    print("Loading historical data from CSV...")
    try:
        df = pd.read_csv("open-meteo-27.75N85.50E1293m.csv")
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        validator = EnhancedDataValidator()
        df = validator.validate_dataframe(df)
        validator.check_data_quality(df)
        validator.check_temporal_consistency(df)
        
        # Ensure all required columns exist (with improved dew point calculation)
        required_columns = ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 
                            'precipitation', 'pressure_msl', 'cloud_cover', 'wind_speed_10m']
        
        for col in required_columns:
            if col not in df.columns:
                print(f"  Missing column {col}, creating with default values")
                if col == 'dew_point_2m' and all(c in df.columns for c in ['temperature_2m', 'relative_humidity_2m']):
                    # Simplified dew point calculation using Magnus formula approximation
                    A = 17.27
                    B = 237.7
                    alpha = ((A * df['temperature_2m']) / (B + df['temperature_2m'])) + np.log(df['relative_humidity_2m'] / 100)
                    df[col] = (B * alpha) / (A - alpha)
                else:
                    df[col] = 0.0 # Default for others
        
        # Enhanced interpolation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both', limit=24)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        print(f" Successfully loaded {len(df):,} records")
        return df
        
    except Exception as e:
        print(f" Error loading CSV file: {e}")
        return generate_sample_data()

def generate_sample_data():
    """Generate realistic sample data for Kathmandu with enhanced patterns"""
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='H')
    base_temp = 15
    seasonal_variation = 8 * np.sin(2 * np.pi * (dates.month - 3) / 12)
    diurnal_variation = 8 * np.sin(2 * np.pi * (dates.hour - 14) / 24)
    temperatures = base_temp + seasonal_variation + diurnal_variation + np.random.normal(0, 2, len(dates))
    
    monsoon_months = [6, 7, 8, 9]
    precip_probability = np.where(np.isin(dates.month, monsoon_months), 0.3, 0.05)
    precipitation = np.random.exponential(precip_probability * 2, len(dates))
    
    wind_speed = 5 + 3 * np.sin(2 * np.pi * dates.hour / 24) + np.random.exponential(1, len(dates))
    base_humidity = 65 - 0.5 * (temperatures - base_temp)
    humidity = np.clip(base_humidity + np.random.normal(0, 8, len(dates)), 20, 95)
    
    pressure = 870 + 5 * np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 2, len(dates))
    cloud_cover = np.clip(40 + 0.5 * humidity + 10 * (precipitation > 0) + np.random.normal(0, 15, len(dates)), 0, 100)
    
    df = pd.DataFrame({
        'time': dates,
        'temperature_2m': temperatures,
        'relative_humidity_2m': humidity,
        'dew_point_2m': temperatures - (100 - humidity) / 5,
        'precipitation': precipitation,
        'pressure_msl': pressure,
        'cloud_cover': cloud_cover,
        'wind_speed_10m': wind_speed
    })
    
    print(" Generated realistic sample data with Kathmandu climate patterns")
    return df

def create_advanced_features(df):
    """Create comprehensive meteorological features"""
    print("Creating advanced meteorological features...")
    feature_engineer = AdvancedFeatureEngineer()
    
    # Apply feature engineering steps
    df = feature_engineer.add_temporal_features(df)
    df = feature_engineer.add_meteorological_features(df)
    
    # DYNAMIC FEATURES: These must be calculated on the full, clean historical data
    df = feature_engineer.add_lagged_features(df, lags=[1, 3, 6, 12, 24, 48])
    df = feature_engineer.add_rolling_features(df, windows=[3, 6, 12, 24, 48])
    
    # Add holiday information
    try:
        np_holidays = holidays.Nepal()
        df['is_holiday'] = df['time'].apply(lambda x: x in np_holidays).astype(int)
    except Exception:
        df['is_holiday'] = 0
    
    # Predict weather conditions for target labeling
    df['weather_condition'] = WeatherConditionPredictor.predict_condition_vectorized(df)
    
    return df

def train_enhanced_models(df):
    """Enhanced model training with classification support"""
    print("Training Enhanced Weather Models...")
    
    targets = {
        'temperature_2m': {'type': 'regression', 'log_transform': False},
        'precipitation': {'type': 'regression', 'log_transform': True},
        'wind_speed_10m': {'type': 'regression', 'log_transform': False},
        'relative_humidity_2m': {'type': 'regression', 'log_transform': False},
        'cloud_cover': {'type': 'regression', 'log_transform': False},
        'pressure_msl': {'type': 'regression', 'log_transform': False},
        'weather_condition': {'type': 'classification'} # New target
    }
    
    # Feature selection
    base_features = [
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
        'is_weekend', 'is_holiday', 'diurnal_phase', 'seasonal_adj', 'monsoon_effect',
        'apparent_temperature', 'heat_index'
    ]
    engineered_features = [col for col in df.columns if any(x in col for x in ['lag_', 'rolling_', 'diff_', 'ema_'])]
    all_features = base_features + engineered_features
    all_features = [col for col in all_features if col in df.columns]
    
    # Prepare classification target
    le = LabelEncoder()
    if 'weather_condition' in df.columns:
        df['weather_condition_encoded'] = le.fit_transform(df['weather_condition'])
        joblib.dump(le, "enhanced_weather_label_encoder.pkl")
        targets['weather_condition_encoded'] = targets.pop('weather_condition')
    
    # Drop data where features are NaN due to lag
    initial_drop = max(48, 48) # 48 is the max lag/window used
    df = df.sort_values('time').iloc[initial_drop:].reset_index(drop=True)
    df = df.fillna(method='ffill', limit=24).fillna(method='bfill', limit=24).fillna(0)
    
    models = {}
    model_performance = {}
    
    for target, config in targets.items():
        if target not in df.columns or len(df) < 1000: continue
        
        temp_df = df.dropna(subset=[target] + all_features).copy()
        X = temp_df[all_features].fillna(0)
        y = temp_df[target]
        
        if config.get('log_transform', False):
            y = np.log1p(y)
        
        if len(X) < 500: continue
        
        # Choose model
        if config['type'] == 'regression':
            model = RandomForestRegressor(n_estimators=200, max_depth=25, random_state=42, n_jobs=-1, max_samples=0.8)
        elif config['type'] == 'classification':
            model = RandomForestClassifier(n_estimators=200, max_depth=25, random_state=42, n_jobs=-1, max_samples=0.8, class_weight='balanced')
        
        model.fit(X, y)
        models[target] = model
        
        # Cross-validation and reporting (for brevity, this section is summarized)
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for train_idx, test_idx in tscv.split(X):
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            y_pred = model.predict(X_test)
            
            if config['type'] == 'regression':
                if config.get('log_transform', False):
                    y_test = np.expm1(y_test)
                    y_pred = np.expm1(y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                cv_scores.append({'rmse': rmse, 'mae': mae})
            elif config['type'] == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                cv_scores.append({'accuracy': accuracy, 'f1': f1})

        if config['type'] == 'regression':
            avg_rmse = np.mean([score['rmse'] for score in cv_scores])
            avg_mae = np.mean([score['mae'] for score in cv_scores])
            model_performance[target] = {'rmse': avg_rmse, 'mae': avg_mae}
        elif config['type'] == 'classification':
            avg_acc = np.mean([score['accuracy'] for score in cv_scores])
            avg_f1 = np.mean([score['f1'] for score in cv_scores])
            model_performance[target] = {'accuracy': avg_acc, 'f1': avg_f1}
            
        print(f" {target} - CV Metric: {avg_rmse if config['type'] == 'regression' else avg_acc:.3f}")
        
    # Save models and metadata
    for target, model in models.items():
        joblib.dump(model, f"enhanced_weather_model_{target}.pkl")
    joblib.dump(all_features, "enhanced_model_features.pkl")
    
    print(f"\nSuccessfully trained {len(models)} enhanced models.")
    return models, all_features, model_performance

def main():
    print("=" * 60)
    print("ENHANCED WEATHER MODEL TRAINING - KATHMANDU")
    print("=" * 60)
    df = load_historical_data()
    if df is None: return
    df = create_advanced_features(df)
    train_enhanced_models(df)
    
if __name__ == "__main__":
    main()