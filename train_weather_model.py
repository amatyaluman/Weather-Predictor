# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import holidays
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("Starting Advanced Weather Model Training...")

class AdvancedFeatureEngineer:
    def add_temporal_features(self, df):
        """Add comprehensive temporal features"""
        df['hour'] = df['time'].dt.hour
        df['month'] = df['time'].dt.month
        df['day_of_year'] = df['time'].dt.dayofyear
        df['weekday'] = df['time'].dt.weekday
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
        
        # Cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        return df
    
    def add_meteorological_features(self, df):
        """Add derived meteorological features based on atmospheric physics"""
        # Diurnal temperature pattern
        df['diurnal_phase'] = 6 * np.sin(2 * np.pi * (df['hour'] - 14) / 24)
        
        # Seasonal adjustments for Nepal climate
        df['seasonal_adj'] = df['month'].apply(self._get_seasonal_adjustment)
        df['monsoon_effect'] = df['month'].apply(lambda x: 0.4 if x in [6,7,8,9] else 0.1)
        
        # Pressure trends and changes - using past only, but for simplicity, remove change to avoid potential issues
        if 'pressure_msl' in df.columns:
            for window in [3, 6, 12, 24]:
                # Removed pressure_change to avoid leakage
                df[f'pressure_rolling_mean_{window}'] = df['pressure_msl'].shift(1).rolling(window=window, min_periods=1).mean()
        
        # Removed temp_humidity_interaction to avoid leakage
        
        return df
    
    def _get_seasonal_adjustment(self, month):
        """Seasonal temperature adjustments for Kathmandu valley"""
        adjustments = {1:-3, 2:-1, 3:2, 4:4, 5:5, 6:3, 7:1, 8:1, 9:2, 10:1, 11:-1, 12:-2}
        return adjustments.get(month, 0)
    
    def add_lagged_features(self, df, lags=[1, 3, 6, 12, 24]):
        """Add lagged features for time series prediction"""
        base_columns = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 
                       'precipitation', 'pressure_msl', 'cloud_cover']
        
        available_columns = [col for col in base_columns if col in df.columns]
        
        for col in available_columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                
                # Removed lagged differences to avoid potential leakage
        
        return df
    
    def add_rolling_features(self, df, windows=[3, 6, 12, 24]):
        """Add rolling statistics for trend analysis using past data only"""
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

class WeatherConditionPredictor:
    @staticmethod
    def predict_condition(row):
        """Predict weather condition based on meteorological parameters"""
        try:
            temp = row.get('temperature_2m', 20)
            precip = row.get('precipitation', 0)
            cloud_cover = row.get('cloud_cover', 0)
            wind_speed = row.get('wind_speed_10m', 0)
            humidity = row.get('relative_humidity_2m', 60)

            # Thunderstorm conditions
            if precip > 8.0 and cloud_cover > 85 and wind_speed > 25:
                return 'Thunderstorm'
            # Heavy rain
            elif precip > 5.0:
                return 'Heavy Rain'
            # Moderate rain
            elif precip > 2.0:
                return 'Moderate Rain'
            # Light rain
            elif precip > 0.5:
                return 'Light Rain'
            # Drizzle
            elif precip > 0.1:
                return 'Drizzle'
            # Snow
            elif temp < 0 and precip > 0.1:
                return 'Snow'
            # Fog
            elif humidity > 90 and cloud_cover > 80 and wind_speed < 5:
                return 'Fog'
            # Cloud conditions
            elif cloud_cover > 80:
                return 'Overcast'
            elif cloud_cover > 60:
                return 'Mostly Cloudy'
            elif cloud_cover > 30:
                return 'Partly Cloudy'
            # Wind conditions
            elif wind_speed > 30:
                return 'Windy'
            elif wind_speed > 20:
                return 'Breezy'
            # Temperature-based conditions
            else:
                if temp > 28:
                    return 'Hot'
                elif temp > 22:
                    return 'Warm'
                elif temp > 15:
                    return 'Pleasant'
                elif temp > 8:
                    return 'Cool'
                else:
                    return 'Cold'
        except Exception as e:
            print(f"Error in condition prediction: {e}")
            return 'Unknown'

class DataValidator:
    @staticmethod
    def validate_dataframe(df):
        """Validate and clean the dataframe"""
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_count = np.isinf(df[numeric_cols]).sum().sum()
        
        if inf_count > 0:
            print(f"Found {inf_count} infinite values, replacing with NaN")
            df = df.replace([np.inf, -np.inf], np.nan)
        
        # Check for large outliers in key metrics
        if 'temperature_2m' in df.columns:
            temp_range = df['temperature_2m'].quantile(0.01), df['temperature_2m'].quantile(0.99)
            outliers = ((df['temperature_2m'] < temp_range[0]) | (df['temperature_2m'] > temp_range[1])).sum()
            if outliers > 0:
                print(f"Found {outliers} temperature outliers, capping values")
                df['temperature_2m'] = df['temperature_2m'].clip(*temp_range)
        
        return df
    
    @staticmethod
    def check_data_quality(df):
        """Comprehensive data quality check"""
        print("\n=== DATA QUALITY REPORT ===")
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['time'].min()} to {df['time'].max()}")
        
        # Check for missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        print("\nMissing values by column:")
        for col, count in missing.items():
            if count > 0:
                print(f"  {col}: {count} ({missing_pct[col]:.1f}%)")
        
        # Basic statistics for key columns
        key_columns = ['temperature_2m', 'precipitation', 'wind_speed_10m', 'relative_humidity_2m']
        available_columns = [col for col in key_columns if col in df.columns]
        
        if available_columns:
            print("\nKey statistics:")
            print(df[available_columns].describe().round(2))
        
        print("=== END QUALITY REPORT ===\n")

def load_historical_data():
    """Load and preprocess historical weather data"""
    print("Loading historical data from CSV...")
    try:
        df = pd.read_csv("open-meteo-27.75N85.50E1293m.csv")
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        
        # Validate data quality
        validator = DataValidator()
        df = validator.validate_dataframe(df)
        validator.check_data_quality(df)
        
        # Ensure all required columns exist
        required_columns = ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 
                           'precipitation', 'pressure_msl', 'cloud_cover', 'wind_speed_10m']
        
        for col in required_columns:
            if col not in df.columns:
                print(f"Warning: Required column '{col}' not found, creating with default values")
                if col == 'temperature_2m':
                    df[col] = 18.0
                elif col == 'relative_humidity_2m':
                    df[col] = 65.0
                elif col == 'precipitation':
                    df[col] = 0.0
                elif col == 'wind_speed_10m':
                    df[col] = 5.0
                elif col == 'pressure_msl':
                    df[col] = 870.0
                elif col == 'cloud_cover':
                    df[col] = 50.0
                else:
                    df[col] = 0.0
        
        # Interpolate missing values
        df[required_columns] = df[required_columns].interpolate(limit_direction='both')
        
        print(f"Successfully loaded {len(df)} records")
        print(f"Date range: {df['time'].min()} to {df['time'].max()}")
        return df
        
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        print("Generating realistic sample data for Kathmandu...")
        return generate_sample_data()

def generate_sample_data():
    """Generate realistic sample data for Kathmandu when CSV is not available"""
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='H')
    
    # Base patterns for Kathmandu (27.7°N, 85.3°E, 1400m elevation)
    base_temp = 15  # Annual average temperature
    
    # Seasonal variation (colder in Jan, warmer in June)
    seasonal_variation = 8 * np.sin(2 * np.pi * (dates.month - 3) / 12)
    
    # Diurnal variation (colder at night, warmer in afternoon)
    diurnal_variation = 8 * np.sin(2 * np.pi * (dates.hour - 14) / 24)
    
    # Realistic temperature with noise
    temperatures = base_temp + seasonal_variation + diurnal_variation + np.random.normal(0, 2, len(dates))
    
    # Precipitation with monsoon pattern (June-September)
    monsoon_months = [6, 7, 8, 9]
    precip_probability = np.where(np.isin(dates.month, monsoon_months), 0.3, 0.05)
    precipitation = np.random.exponential(precip_probability * 2, len(dates))
    
    # Wind speed with diurnal pattern
    wind_speed = 5 + 3 * np.sin(2 * np.pi * dates.hour / 24) + np.random.exponential(1, len(dates))
    
    # Humidity with inverse relationship to temperature
    base_humidity = 65 - 0.5 * (temperatures - base_temp)
    humidity = np.clip(base_humidity + np.random.normal(0, 8, len(dates)), 20, 95)
    
    # Pressure at 1400m elevation (~870 hPa average)
    pressure = 870 + 5 * np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 2, len(dates))
    
    # Cloud cover related to precipitation and humidity
    cloud_cover = np.clip(40 + 0.5 * humidity + 10 * (precipitation > 0) + np.random.normal(0, 15, len(dates)), 0, 100)
    
    df = pd.DataFrame({
        'time': dates,
        'temperature_2m': temperatures,
        'relative_humidity_2m': humidity,
        'dew_point_2m': temperatures - (100 - humidity) / 5,  # Approximation
        'precipitation': precipitation,
        'pressure_msl': pressure,
        'cloud_cover': cloud_cover,
        'wind_speed_10m': wind_speed
    })
    
    print("Generated sample data with realistic Kathmandu patterns")
    return df

def create_advanced_features(df):
    """Create comprehensive features for model training"""
    print("Creating advanced meteorological features...")
    
    feature_engineer = AdvancedFeatureEngineer()
    
    # Add temporal features
    df = feature_engineer.add_temporal_features(df)
    
    # Add meteorological features
    df = feature_engineer.add_meteorological_features(df)
    
    # Add lagged features
    df = feature_engineer.add_lagged_features(df, lags=[1, 3, 6, 12, 24])
    
    # Add rolling features
    df = feature_engineer.add_rolling_features(df, windows=[3, 6, 12, 24])
    
    # Add holiday information
    try:
        np_holidays = holidays.Nepal()
        df['is_holiday'] = df['time'].apply(lambda x: x in np_holidays).astype(int)
    except Exception as e:
        print(f"Could not load holiday data: {e}")
        df['is_holiday'] = 0
    
    # Add weather condition labels
    condition_predictor = WeatherConditionPredictor()
    df['weather_condition'] = df.apply(condition_predictor.predict_condition, axis=1)
    
    print(f"Created features. Total columns: {len(df.columns)}")
    return df

def train_advanced_models(df):
    """Train machine learning models for weather prediction"""
    print("Training Advanced Weather Models...")
    
    # Define prediction targets
    targets = {
        'temperature_2m': 'regression',
        'precipitation': 'regression', 
        'wind_speed_10m': 'regression',
        'relative_humidity_2m': 'regression',
        'cloud_cover': 'regression',
        'pressure_msl': 'regression'  # Added to predict pressure
    }
    
    # Base feature set (temporal and derived features)
    base_features = [
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
        'is_weekend', 'is_holiday', 'diurnal_phase', 'seasonal_adj', 'monsoon_effect'
    ]
    
    # Add all engineered features
    all_features = base_features + [col for col in df.columns if any(x in col for x in [
        'lag_', 'rolling_'
    ])]  # Removed 'change_', 'diff_', 'interaction'
    
    # Filter to only existing columns
    all_features = [col for col in all_features if col in df.columns]
    
    print(f"Using {len(all_features)} features for training")
    print("Feature examples:", all_features[:10])
    
    models = {}
    feature_importance = {}
    
    for target in targets.keys():
        print(f"\n--- Training model for {target} ---")
        
        if target not in df.columns:
            print(f"Skipping {target}, not in data")
            continue
            
        # Prepare data
        temp_df = df.dropna(subset=all_features + [target]).copy()
        
        if len(temp_df) < 1000:
            print(f"Skipping {target}, insufficient samples: {len(temp_df)}")
            continue
            
        X = temp_df[all_features].replace([np.inf, -np.inf], np.nan)
        y = temp_df[target].replace([np.inf, -np.inf], np.nan)
        
        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 500:
            print(f"Skipping {target}, valid rows: {len(X)}")
            continue
            
        print(f"Training on {len(X)} samples...")
        
        # Train main model
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            bootstrap=True
        )
        
        model.fit(X, y)
        models[target] = model
        
        # Store feature importance
        feature_importance[target] = dict(zip(all_features, model.feature_importances_))
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            cv_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
            cv_model.fit(X_train, y_train)
            y_pred = cv_model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            cv_scores.append({'rmse': rmse, 'mae': mae})
        
        avg_rmse = np.mean([score['rmse'] for score in cv_scores])
        avg_mae = np.mean([score['mae'] for score in cv_scores])
        
        print(f" Cross-validation RMSE: {avg_rmse:.3f}")
        print(f" Cross-validation MAE: {avg_mae:.3f}")
        
        # Show top features
        top_features = sorted(feature_importance[target].items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top features: {[f[0] for f in top_features]}")
    
    # Save models and metadata
    if models:
        joblib.dump(models, "weather_models.pkl")
        joblib.dump(all_features, "model_features.pkl")
        
        # Save feature importance
        metadata = {
            'feature_importance': feature_importance,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'training_samples': len(df),
            'features_used': all_features
        }
        joblib.dump(metadata, "model_metadata.pkl")
        
        print(f"\n Successfully trained {len(models)} models")
        print(f" Models saved: weather_models.pkl")
        print(f" Features saved: model_features.pkl")
        print(f" Metadata saved: model_metadata.pkl")
        
    else:
        print(" No models were trained successfully")
        
    return models, all_features

def main():
    print("=" * 60)
    print("ADVANCED WEATHER MODEL TRAINING - KATHMANDU")
    print("=" * 60)
    
    # Load and prepare data
    df = load_historical_data()
    if df is None:
        print(" Failed to load data")
        return
    
    # Create advanced features
    df = create_advanced_features(df)
    
    # Remove rows with too many NaN values
    initial_count = len(df)
    df = df.dropna()
    final_count = len(df)
    
    print(f"Data cleaning: {initial_count} → {final_count} records ({((initial_count-final_count)/initial_count*100):.1f}% removed)")
    
    if len(df) < 1000:
        print(" Insufficient data for training (need at least 1000 records)")
        return
    
    # Train models
    models, features = train_advanced_models(df)
    
    if models:
        print("\n Model training completed successfully!")
        print(f" Trained models for: {list(models.keys())}")
    else:
        print("\n Model training failed")

if __name__ == "__main__":
    main()