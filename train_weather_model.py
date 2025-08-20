import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# -------------------- LOAD DATA --------------------
# Define column names for hourly data (22 columns)
columns = [
    'location_id', 'time', 'temperature_2m (°C)', 'relative_humidity_2m (%)', 
    'apparent_temperature (°C)', 'dew_point_2m (°C)', 'precipitation (mm)', 
    'wind_gusts_10m (km/h)', 'visibility (m)', 'cape (J/kg)', 'lightning_potential (J/kg)', 
    'is_day ()', 'rain (mm)', 'snowfall (cm)', 'snowfall_height (m)', 
    'freezing_level_height (m)', 'sunshine_duration (s)', 'weather_code (wmo code)', 
    'wind_speed_10m (km/h)', 'wind_speed_80m (km/h)', 'wind_direction_10m (°)', 
    'wind_direction_80m (°)'
]

# Read only the hourly data section (rows 5 to 108406, accounting for 4 metadata rows)
df = pd.read_csv("KTMLTP.csv", skiprows=4, nrows=108402, names=columns, header=None)

# Filter for location 0 (hourly data)
df = df[df['location_id'] == 0]

# -------------------- HANDLE DATE FORMAT --------------------
# Specify the date format for hourly data (adjust based on actual format)
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# Check for any parsing errors (rows where time is NaT)
if df['time'].isna().any():
    print(f"Warning: {df['time'].isna().sum()} rows have invalid dates in the time column")
    # Optionally drop or handle invalid dates
    df = df.dropna(subset=['time'])

# -------------------- RENAME COLUMNS TO REMOVE UNITS --------------------
rename_dict = {
    'temperature_2m (°C)': 'temperature_2m',
    'relative_humidity_2m (%)': 'relative_humidity_2m',
    'apparent_temperature (°C)': 'apparent_temperature',
    'dew_point_2m (°C)': 'dew_point_2m',
    'precipitation (mm)': 'precipitation',
    'wind_gusts_10m (km/h)': 'wind_gusts_10m',
    'visibility (m)': 'visibility',
    'cape (J/kg)': 'cape',
    'lightning_potential (J/kg)': 'lightning_potential',
    'is_day ()': 'is_day',
    'rain (mm)': 'rain',
    'snowfall (cm)': 'snowfall',
    'snowfall_height (m)': 'snowfall_height',
    'freezing_level_height (m)': 'freezing_level_height',
    'sunshine_duration (s)': 'sunshine_duration',
    'weather_code (wmo code)': 'weather_code',
    'wind_speed_10m (km/h)': 'wind_speed_10m',
    'wind_speed_80m (km/h)': 'wind_speed_80m',
    'wind_direction_10m (°)': 'wind_direction_10m',
    'wind_direction_80m (°)': 'wind_direction_80m'
}
df = df.rename(columns=rename_dict)

# -------------------- FEATURE ENGINEERING --------------------
df['hour'] = df['time'].dt.hour
df['month'] = df['time'].dt.month
df['day_of_week'] = df['time'].dt.dayofweek

# Cyclical features for time
df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
df['month_cos'] = np.cos(2 * np.pi * df['month']/12)

# Create weather type based on weather_code
bins = [0, 1, 3, 50, 70, 100]
labels = ['clear', 'cloudy', 'fog', 'rain', 'storm']
df['weather_type'] = pd.cut(df['weather_code'], bins=bins, labels=labels, include_lowest=True)
df = pd.get_dummies(df, columns=['weather_type'])

# -------------------- FEATURES AND TARGET --------------------
feature_cols = [
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'hour', 'month', 'day_of_week',
    'relative_humidity_2m', 'wind_speed_10m', 'wind_direction_10m'
]

# Add weather dummy columns
weather_dummies = [col for col in df.columns if col.startswith('weather_type_')]
feature_cols.extend(weather_dummies)

X = df[feature_cols]
y = df['temperature_2m']

# Handle NaN values
X = X.fillna(0)
y = y.fillna(0)

# -------------------- SPLIT DATA --------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------- TRAIN RANDOM FOREST --------------------
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# -------------------- EVALUATE --------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.2f}")

# -------------------- SAVE MODEL --------------------
joblib.dump(model, "optimized_weather_model.pkl", compress=3)
print("Optimized model saved as optimized_weather_model.pkl")