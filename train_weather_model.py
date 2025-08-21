import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# -------------------- CSV FILE --------------------
csv_file = "open-meteo-27.75N85.50E1293mNew.csv"

# -------------------- READ CSV --------------------
print(f"Reading CSV: {csv_file}")
df = pd.read_csv(csv_file)
df['time'] = pd.to_datetime(df['time'], errors='coerce')

# -------------------- FEATURE ENGINEERING --------------------
df['hour'] = df['time'].dt.hour
df['day_of_week'] = df['time'].dt.dayofweek
df['month'] = df['time'].dt.month
df['year'] = df['time'].dt.year
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

# Convert all other columns to numeric
numeric_cols = ['temperature_2m','relative_humidity_2m','dew_point_2m',
                'precipitation','pressure_msl','cloud_cover','visibility',
                'wind_speed_10m','wind_direction_10m','wind_gusts_10m']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"Data shape after preprocessing: {df.shape}")

# -------------------- DEFINE TARGETS --------------------
targets = ['temperature_2m','relative_humidity_2m','dew_point_2m',
           'precipitation','wind_speed_10m','wind_direction_10m','wind_gusts_10m']

available_targets = [t for t in targets if t in df.columns]
print(f"Training targets: {available_targets}")

# -------------------- PREPARE FEATURES --------------------
feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in available_targets]

for col in feature_cols:
    df[col].fillna(df[col].median(), inplace=True)

print(f"Using {len(feature_cols)} feature columns: {feature_cols}")

# -------------------- TRAIN MODELS --------------------
model_data = {
    'feature_columns': feature_cols,
    'target_columns': available_targets,
    'models': {},
    'performance': {}
}

for target in available_targets:
    print(f"\nTraining model for: {target}")
    y = df[target].copy()
    valid_mask = y.notna()
    if valid_mask.sum() < 20:
        print(f"  Not enough data: {valid_mask.sum()} samples")
        continue

    X_valid = df[feature_cols][valid_mask]
    y_valid = y[valid_mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X_valid, y_valid, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    model_data['models'][target] = model
    model_data['performance'][target] = {
        'rmse': rmse,
        'mae': mae,
        'samples': len(X_valid)
    }

# -------------------- SAVE MODEL --------------------
model_filename = "weather_prediction_hourly_model.pkl"
joblib.dump(model_data, model_filename)
print(f"\n✓ Model saved: {model_filename}")
print(f"Trained models: {list(model_data['models'].keys())}")
