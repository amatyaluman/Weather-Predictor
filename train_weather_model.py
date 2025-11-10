# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import joblib
import holidays
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("Starting Advanced Weather Model Training...")

class WeatherConditionPredictor:
    @staticmethod
    def predict_condition(row):
        try:
            temp = row.get('temperature_2m', 20)
            precip = row.get('precipitation', 0)
            cloud_cover = row.get('cloud_cover', 0)
            wind_speed = row.get('wind_speed_10m', 0)
            humidity = row.get('relative_humidity_2m', 60)
            if precip > 8.0 and cloud_cover > 85 and wind_speed > 25:
                return 'Thunderstorm'
            elif precip > 5.0:
                return 'Heavy Rain'
            elif precip > 2.0:
                return 'Moderate Rain'
            elif precip > 0.5:
                return 'Light Rain'
            elif precip > 0.1:
                return 'Drizzle'
            elif temp < 0 and precip > 0.1:
                return 'Snow'
            elif humidity > 90 and cloud_cover > 80 and wind_speed < 5:
                return 'Fog'
            elif cloud_cover > 80:
                return 'Overcast'
            elif cloud_cover > 60:
                return 'Mostly Cloudy'
            elif cloud_cover > 30:
                return 'Partly Cloudy'
            elif wind_speed > 30:
                return 'Windy'
            elif wind_speed > 20:
                return 'Breezy'
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
        except Exception:
            return 'Unknown'

class AdvancedFeatureEngineer:
    def add_seasonal_features(self, df):
        df['diurnal_temp'] = 8 * np.sin(2 * np.pi * (df['hour'] - 6) / 24)
        df['seasonal_adj'] = df['month'].apply(self._get_seasonal_adjustment)
        df['monsoon_effect'] = df['month'].apply(lambda x: -3 if x in [6,7,8,9] else 2)
        return df
    def _get_seasonal_adjustment(self, month):
        adjustments = {1:-5,2:-3,3:2,4:5,5:6,6:4,7:2,8:2,9:3,10:2,11:-1,12:-4}
        return adjustments.get(month, 0)
    def add_advanced_lags(self, df):
        for lag in [1, 3, 6, 12, 24]:
            df[f'temp_lag_{lag}_adj'] = df['temperature_2m'].shift(lag) + df['diurnal_temp'] - df['diurnal_temp'].shift(lag)
        if 'pressure_msl' in df.columns:
            for window in [3, 6, 12]:
                df[f'pressure_change_{window}h'] = df['pressure_msl'] - df['pressure_msl'].shift(window)
        return df

class DataValidator:
    @staticmethod
    def validate_dataframe(df):
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            df = df.replace([np.inf, -np.inf], np.nan)
        return df

def load_historical_data():
    print("Loading historical data from CSV...")
    try:
        df = pd.read_csv("open-meteo-27.75N85.50E1293m.csv")
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        validator = DataValidator()
        df = validator.validate_dataframe(df)
        need = ['temperature_2m','relative_humidity_2m','dew_point_2m','precipitation','pressure_msl','cloud_cover','wind_speed_10m']
        for c in need:
            if c not in df.columns:
                df[c] = np.nan
        df[need] = df[need].interpolate(limit_direction='both')
        print(f"Successfully loaded {len(df)} records")
        print(f"Date range: {df['time'].min()} to {df['time'].max()}")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def create_realistic_features(df):
    print("Creating realistic features...")
    fe = AdvancedFeatureEngineer()
    df['hour'] = df['time'].dt.hour
    df['day'] = df['time'].dt.day
    df['month'] = df['time'].dt.month
    df['weekday'] = df['time'].dt.weekday
    df['day_of_year'] = df['time'].dt.dayofyear
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    try:
        np_holidays = holidays.Nepal()
        df['is_holiday'] = df['time'].apply(lambda x: x in np_holidays).astype(int)
    except Exception:
        df['is_holiday'] = 0
    df = fe.add_seasonal_features(df)
    features_base = ['temperature_2m','relative_humidity_2m','wind_speed_10m','precipitation','dew_point_2m','cloud_cover','pressure_msl']
    for col in features_base:
        for lag in [1, 3, 6, 12, 24]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    df = fe.add_advanced_lags(df)
    windows = [3, 6, 12, 24]
    for window in windows:
        for col in ['temperature_2m','relative_humidity_2m','wind_speed_10m','pressure_msl']:
            if col in df.columns:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
    return df

def train_accurate_models(df):
    print("Training Accurate Weather Models...")
    targets = {'temperature_2m':'regression','precipitation':'regression','wind_speed_10m':'regression','relative_humidity_2m':'regression','cloud_cover':'regression'}
    base_features = ['hour_sin','hour_cos','month_sin','month_cos','is_weekend','is_holiday','day_of_year','diurnal_temp','seasonal_adj','monsoon_effect']
    all_features = base_features + [c for c in df.columns if any(x in c for x in ['lag_','rolling_','change_','adj']) and c not in base_features]
    all_features = [c for c in all_features if c in df.columns]
    models = {}
    for target in targets.keys():
        print(f"Training model for {target}...")
        if target not in df.columns:
            print(f"Skipping {target}, not in data")
            continue
        temp_df = df.dropna(subset=all_features + [target]).copy()
        if len(temp_df) < 1000:
            print(f"Skipping {target}, insufficient samples: {len(temp_df)}")
            continue
        X = temp_df[all_features].replace([np.inf,-np.inf], np.nan)
        y = temp_df[target].replace([np.inf,-np.inf], np.nan)
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        if len(X) < 500:
            print(f"Skipping {target}, valid rows: {len(X)}")
            continue
        model = RandomForestRegressor(n_estimators=300,max_depth=22,min_samples_split=3,min_samples_leaf=1,max_features='sqrt',random_state=42,n_jobs=-1,bootstrap=True)
        model.fit(X, y)
        models[target] = model
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for tr, te in tscv.split(X):
            Xm_tr, Xm_te = X.iloc[tr], X.iloc[te]
            ym_tr, ym_te = y.iloc[tr], y.iloc[te]
            m = RandomForestRegressor(n_estimators=150,max_depth=18,random_state=42,n_jobs=-1)
            m.fit(Xm_tr, ym_tr)
            yp = m.predict(Xm_te)
            rmse = float(np.sqrt(mean_squared_error(ym_te, yp)))
            scores.append(rmse)
        print(f"  Cross-validation RMSE: {np.mean(scores):.3f}")
    if models:
        joblib.dump(models, "weather_models.pkl")
        joblib.dump(all_features, "model_features.pkl")
        print(f"Successfully trained {len(models)} models")
    return models, all_features

def main():
    print("=" * 50)
    print("ACCURATE WEATHER MODEL TRAINING")
    print("=" * 50)
    df = load_historical_data()
    if df is None:
        return
    df = create_realistic_features(df)
    df = df.dropna()
    print(f"Final training dataset: {len(df)} records")
    if len(df) < 1000:
        print("Insufficient data for training")
        return
    models, features = train_accurate_models(df)
    if models:
        print("Model training completed successfully!")
    else:
        print("Model training failed")

if __name__ == "__main__":
    main()
