# train_weather_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("Kathmandu AI Weather — Training Trend-Only Model (Real Data Only)")
print("="*70)

# Load CSV
try:
    df = pd.read_csv("open-meteo-27.75N85.50E1293m.csv")
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    print(f"Loaded {len(df):,} real hourly records")
    print(f"From: {df['time'].min().date()} → To: {df['time'].max().date()}")
except:
    raise FileNotFoundError("open-meteo-27.75N85.50E1293m.csv not found!")

# Time-based features only (NO lag features = works after any gap)
df['hour'] = df['time'].dt.hour
df['month'] = df['time'].dt.month
df['doy'] = df['time'].dt.dayofyear
df['year'] = df['time'].dt.year

df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['doy_sin'] = np.sin(2 * np.pi * df['doy'] / 365)
df['doy_cos'] = np.cos(2 * np.pi * df['doy'] / 365)

df['is_monsoon'] = df['month'].isin([6,7,8,9]).astype(int)
df['year_progress'] = df['doy'] / 365
df['days_since_2020'] = (df['time'] - pd.Timestamp("2020-01-01")).dt.days

# Final features — works even in 2030
feature_cols = [
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'doy_sin', 'doy_cos', 'is_monsoon', 'year_progress', 'days_since_2020'
]

targets = [
    'temperature_2m', 'relative_humidity_2m', 'precipitation',
    'wind_speed_10m', 'cloud_cover', 'pressure_msl'
]

models = {}
scalers = {}

print("\nTraining 6 independent models...")
for target in targets:
    print(f"   Training {target}...", end="")
    
    X = df[feature_cols]
    y = df[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestRegressor(
        n_estimators=600,
        max_depth=30,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled, y)
    
    pred = model.predict(X_scaled)
    mae = mean_absolute_error(y, pred)
    
    models[target] = {'model': model, 'scaler': scaler}
    scalers[target] = scaler
    
    print(f" Done (MAE: {mae:.3f})")

# Save
joblib.dump({
    'models': models,
    'feature_cols': feature_cols,
    'created': datetime.now().strftime("%Y-%m-%d %H:%M"),
    'data_end': df['time'].max().strftime("%Y-%m-%d")
}, "kathmandu_trend_model.pkl")

print("\nMODEL SAVED: kathmandu_trend_model.pkl")
print("Now run: streamlit run weather_dashboard.py")