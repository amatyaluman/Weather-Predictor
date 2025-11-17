import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
        """Enhanced meteorological features"""
        # Diurnal and seasonal patterns
        df['diurnal_phase'] = 6 * np.sin(2 * np.pi * (df['hour'] - 14) / 24)
        df['seasonal_adj'] = df['month'].apply(self._get_seasonal_adjustment)
        df['monsoon_effect'] = df['month'].apply(lambda x: 0.4 if x in [6,7,8,9] else 0.1)
        
        # Pressure features
        if 'pressure_msl' in df.columns:
            for window in [3, 6, 12, 24]:
                df[f'pressure_rolling_mean_{window}'] = (
                    df['pressure_msl'].shift(1).rolling(window=window, min_periods=1).mean()
                )
                df[f'pressure_trend_{window}'] = (
                    df['pressure_msl'].diff().rolling(window=window, min_periods=1).mean()
                )
        
        # Temperature-humidity interactions
        if all(col in df.columns for col in ['temperature_2m', 'relative_humidity_2m']):
            df['apparent_temperature'] = df['temperature_2m'] + 0.33 * (df['relative_humidity_2m'] / 100 * 6.105 * 
                                       np.exp(17.27 * df['temperature_2m'] / (237.7 + df['temperature_2m']))) - 4
            df['heat_index'] = self.calculate_heat_index(df['temperature_2m'], df['relative_humidity_2m'])
        
        return df

    def calculate_heat_index(self, temperature, humidity):
        """Calculate heat index based on temperature and humidity"""
        # Simplified heat index calculation
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

        default_temp = pd.Series('Unknown', index=df.index)

        temp_hot = temp > 28
        temp_warm = temp > 22
        temp_pleasant = temp > 15
        temp_cool = temp > 8
        temp_cold = ~temp_cool

        temp_choices = ['Hot', 'Warm', 'Pleasant', 'Cool', 'Cold']
        temp_conditions = [temp_hot, temp_warm, temp_pleasant, temp_cool, temp_cold]

        result = np.select(conditions, choices, default='__TEMP_CHECK__')
        temp_result = np.select(temp_conditions, temp_choices, default='Unknown')
        mask_temp = result == '__TEMP_CHECK__'
        result[mask_temp] = temp_result[mask_temp]
        return pd.Series(result, index=df.index)

class EnhancedDataValidator:
    @staticmethod
    def validate_dataframe(df):
        """Enhanced data validation with more robust outlier detection"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Handle infinite values
        inf_mask = np.isinf(df[numeric_cols])
        inf_count = inf_mask.sum().sum()
        if inf_count > 0:
            print(f"Replacing {inf_count} infinite values with NaN")
            df = df.replace([np.inf, -np.inf], np.nan)
        
        # Handle outliers using IQR method for all numeric columns
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers > 0:
                    print(f"Clipping {outliers} outliers in {col}")
                    df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df

    @staticmethod
    def check_data_quality(df):
        """Comprehensive data quality assessment"""
        print("\n=== ENHANCED DATA QUALITY REPORT ===")
        print(f"Total records: {len(df):,}")
        print(f"Date range: {df['time'].min()} to {df['time'].max()}")
        print(f"Duration: {(df['time'].max() - df['time'].min()).days} days")
        
        # Missing values analysis
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        print("\nMissing values by column:")
        for col, count in missing.items():
            if count > 0:
                print(f"  {col:25}: {count:6} ({missing_pct[col]:5.1f}%)")
        
        # Key statistics for important columns
        key_columns = ['temperature_2m', 'precipitation', 'wind_speed_10m', 
                      'relative_humidity_2m', 'pressure_msl', 'cloud_cover']
        available_columns = [col for col in key_columns if col in df.columns]
        
        if available_columns:
            print("\nKey statistics:")
            stats = df[available_columns].describe().round(2)
            print(stats)
        
        # Data completeness score
        completeness = (1 - missing_pct / 100).mean()
        print(f"\nOverall data completeness: {completeness:.1%}")
        print("=== END ENHANCED QUALITY REPORT ===\n")

    @staticmethod
    def check_temporal_consistency(df):
        """Check for temporal gaps in data"""
        if 'time' not in df.columns:
            print(" Time column not found for temporal consistency check")
            return False
            
        time_diff = df['time'].diff()
        expected_freq = pd.Timedelta(hours=1)
        gaps = time_diff[time_diff > expected_freq * 1.1]  # 10% tolerance
        
        if len(gaps) > 0:
            print(f"  Found {len(gaps)} temporal gaps in data")
            for gap in gaps.head(3):  # Show first 3 gaps
                print(f"  Gap of {gap}")
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
        
        # Ensure all required columns exist
        required_columns = ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 
                           'precipitation', 'pressure_msl', 'cloud_cover', 'wind_speed_10m']
        
        for col in required_columns:
            if col not in df.columns:
                print(f"  Missing column {col}, creating with default values")
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
                elif col == 'dew_point_2m':
                    # Calculate dew point if we have temp and humidity
                    if all(c in df.columns for c in ['temperature_2m', 'relative_humidity_2m']):
                        df[col] = df['temperature_2m'] - (100 - df['relative_humidity_2m']) / 5
                    else:
                        df[col] = 10.0
                else:
                    df[col] = 0.0
        
        # Enhanced interpolation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both', limit=24)
        
        # Fill any remaining NaNs
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        print(f" Successfully loaded {len(df):,} records")
        print(f" Date range: {df['time'].min()} to {df['time'].max()}")
        return df
        
    except Exception as e:
        print(f" Error loading CSV file: {e}")
        print(" Generating realistic sample data for Kathmandu...")
        return generate_sample_data()

def generate_sample_data():
    """Generate realistic sample data for Kathmandu with enhanced patterns"""
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='H')
    
    # Base temperature with seasonal and diurnal variations
    base_temp = 15
    seasonal_variation = 8 * np.sin(2 * np.pi * (dates.month - 3) / 12)
    diurnal_variation = 8 * np.sin(2 * np.pi * (dates.hour - 14) / 24)
    temperatures = base_temp + seasonal_variation + diurnal_variation + np.random.normal(0, 2, len(dates))
    
    # Precipitation with monsoon pattern
    monsoon_months = [6, 7, 8, 9]
    precip_probability = np.where(np.isin(dates.month, monsoon_months), 0.3, 0.05)
    precipitation = np.random.exponential(precip_probability * 2, len(dates))
    
    # Wind speed with diurnal pattern
    wind_speed = 5 + 3 * np.sin(2 * np.pi * dates.hour / 24) + np.random.exponential(1, len(dates))
    
    # Humidity inversely related to temperature
    base_humidity = 65 - 0.5 * (temperatures - base_temp)
    humidity = np.clip(base_humidity + np.random.normal(0, 8, len(dates)), 20, 95)
    
    # Pressure with slight diurnal variation
    pressure = 870 + 5 * np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 2, len(dates))
    
    # Cloud cover related to humidity and precipitation
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
    print(f" Sample size: {len(df):,} records")
    return df

def create_advanced_features(df):
    """Create comprehensive meteorological features"""
    print("Creating advanced meteorological features...")
    feature_engineer = AdvancedFeatureEngineer()
    
    # Apply feature engineering steps
    df = feature_engineer.add_temporal_features(df)
    df = feature_engineer.add_meteorological_features(df)
    df = feature_engineer.add_lagged_features(df, lags=[1, 3, 6, 12, 24, 48])
    df = feature_engineer.add_rolling_features(df, windows=[3, 6, 12, 24, 48])
    
    # Add holiday information
    try:
        np_holidays = holidays.Nepal()
        df['is_holiday'] = df['time'].apply(lambda x: x in np_holidays).astype(int)
        print(f" Added holiday information ({df['is_holiday'].sum()} holidays)")
    except Exception as e:
        print(f" Could not load holiday data: {e}")
        df['is_holiday'] = 0
    
    # Predict weather conditions
    df['weather_condition'] = WeatherConditionPredictor.predict_condition_vectorized(df)
    
    print(f" Created {len(df.columns)} features total")
    print(f" Feature categories: temporal, lagged, rolling, meteorological, holidays")
    
    return df

def train_enhanced_models(df):
    """Enhanced model training with hyperparameter tuning and better evaluation"""
    print("Training Enhanced Weather Models...")
    
    targets = {
        'temperature_2m': {'type': 'regression', 'priority': 1},
        'precipitation': {'type': 'regression', 'priority': 1, 'log_transform': True},
        'wind_speed_10m': {'type': 'regression', 'priority': 2},
        'relative_humidity_2m': {'type': 'regression', 'priority': 2},
        'cloud_cover': {'type': 'regression', 'priority': 3},
        'pressure_msl': {'type': 'regression', 'priority': 3}
    }
    
    # Feature selection
    base_features = [
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
        'is_weekend', 'is_holiday', 'diurnal_phase', 'seasonal_adj', 'monsoon_effect'
    ]
    
    engineered_features = [col for col in df.columns if any(x in col for x in ['lag_', 'rolling_', 'diff_', 'ema_', 'apparent_temperature', 'heat_index'])]
    all_features = base_features + engineered_features
    all_features = [col for col in all_features if col in df.columns]
    
    print(f" Using {len(all_features)} features for training")
    print(f" Feature examples: {all_features[:8]}")
    
    # Calculate maximum lag for initial data drop
    max_lag = 0
    for col in engineered_features:
        if 'lag_' in col:
            try:
                lag_val = int(col.split('_')[-1])
                max_lag = max(max_lag, lag_val)
            except ValueError:
                continue
    
    initial_drop = max(48, max_lag)
    df = df.sort_values('time').reset_index(drop=True)
    if initial_drop > 0:
        print(f" Dropping first {initial_drop} records due to lag features")
        df = df.iloc[initial_drop:].reset_index(drop=True)
    
    # Enhanced imputation
    df = df.fillna(method='ffill', limit=24).fillna(method='bfill', limit=24).fillna(0)
    
    models = {}
    model_performance = {}
    feature_importance = {}
    
    for target, config in targets.items():
        print(f"\n--- Training model for {target} ---")
        
        if target not in df.columns:
            print(f" Skipping {target}, not in data")
            continue
            
        temp_df = df.dropna(subset=[target] + all_features).copy()
        
        if len(temp_df) < 1000:
            print(f" Skipping {target}, insufficient samples: {len(temp_df)}")
            continue
        
        # Prepare features and target
        X = temp_df[all_features].replace([np.inf, -np.inf], np.nan)
        y = temp_df[target].replace([np.inf, -np.inf], np.nan)
        
        # Apply transformations if specified
        original_y = y.copy()
        if config.get('log_transform', False):
            y = np.log1p(y)  # log(1 + x) to handle zeros
        
        # Remove rows with any NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 500:
            print(f" Skipping {target}, valid rows: {len(X)}")
            continue
            
        print(f" Training on {len(X):,} samples...")
        
        # Enhanced model with better hyperparameters
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=25,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            max_samples=0.8
        )
        
        model.fit(X, y)
        models[target] = model
        feature_importance[target] = dict(zip(all_features, model.feature_importances_))
        
        # Enhanced cross-validation with more metrics
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            cv_model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=20, 
                random_state=42, 
                n_jobs=-1
            )
            cv_model.fit(X_train, y_train)
            y_pred = cv_model.predict(X_test)
            
            # Transform back if log transformed
            if config.get('log_transform', False):
                y_test_orig = np.expm1(y_test)
                y_pred_orig = np.expm1(y_pred)
            else:
                y_test_orig = y_test
                y_pred_orig = y_pred
            
            rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
            mae = mean_absolute_error(y_test_orig, y_pred_orig)
            r2 = r2_score(y_test_orig, y_pred_orig)
            
            cv_scores.append({'rmse': rmse, 'mae': mae, 'r2': r2})
        
        # Calculate average metrics
        avg_rmse = np.mean([score['rmse'] for score in cv_scores])
        avg_mae = np.mean([score['mae'] for score in cv_scores])
        avg_r2 = np.mean([score['r2'] for score in cv_scores])
        
        print(f" Cross-validation RMSE: {avg_rmse:.3f}")
        print(f" Cross-validation MAE: {avg_mae:.3f}")
        print(f"  Cross-validation R²: {avg_r2:.3f}")
        
        model_performance[target] = {
            'rmse': avg_rmse,
            'mae': avg_mae,
            'r2': avg_r2,
            'training_samples': len(X)
        }
        
        # Feature importance analysis
        top_features = sorted(feature_importance[target].items(), 
                            key=lambda x: x[1], reverse=True)[:8]
        print(f"  Top features:")
        for feature, importance in top_features:
            print(f"    {feature:30}: {importance:.4f}")
    
    # Save models and metadata
    if models:
        try:
            # Save individual models
            for target, model in models.items():
                joblib.dump(model, f"enhanced_weather_model_{target}.pkl")
            
            # Save comprehensive metadata
            metadata = {
                'feature_importance': feature_importance,
                'model_performance': model_performance,
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'training_samples': len(df),
                'features_used': all_features,
                'targets_trained': list(models.keys()),
                'data_range': {
                    'start': df['time'].min().strftime('%Y-%m-%d'),
                    'end': df['time'].max().strftime('%Y-%m-%d')
                }
            }
            
            joblib.dump(all_features, "enhanced_model_features.pkl")
            joblib.dump(metadata, "enhanced_model_metadata.pkl")
            
            print(f"\n Successfully trained {len(models)} enhanced models")
            print(" Models saved: enhanced_weather_model_<target>.pkl")
            print(" Performance report saved in metadata")
            
            # Print summary
            print("\n MODEL PERFORMANCE SUMMARY:")
            print("=" * 60)
            for target, perf in model_performance.items():
                print(f"{target:25} | RMSE: {perf['rmse']:6.3f} | MAE: {perf['mae']:6.3f} | R²: {perf['r2']:6.3f}")
            
        except Exception as e:
            print(f" Error saving models: {e}")
    else:
        print(" No models were trained successfully")
    
    return models, all_features, model_performance

def main():
    print("=" * 60)
    print("ENHANCED WEATHER MODEL TRAINING - KATHMANDU")
    print("=" * 60)
    
    # Load and validate data
    df = load_historical_data()
    if df is None:
        print(" Failed to load data")
        return
    
    # Create features and train models
    df = create_advanced_features(df)
    models, features, performance = train_enhanced_models(df)
    
    if models:
        print(f"\n Training completed! {len(models)} models ready for deployment")
        print(f"Data used: {len(df):,} records")
        print(f" Features: {len(features)} engineered features")
    else:
        print("\n Training failed")

if __name__ == "__main__":
    main()