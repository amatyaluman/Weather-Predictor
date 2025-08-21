import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# -------------------- LOAD DATA --------------------
df = pd.read_csv("open-meteo-27.75N85.50E1293mNew.csv")

# Convert 'time' column to datetime
df['time'] = pd.to_datetime(df['time'])

# -------------------- FEATURE ENGINEERING --------------------
# Extract time-based features
df['hour'] = df['time'].dt.hour
df['day'] = df['time'].dt.day
df['month'] = df['time'].dt.month
df['weekday'] = df['time'].dt.weekday

# Optional: Shift target for forecasting next hour
df['temperature_next_hour'] = df['temperature_2m'].shift(-1)

# Drop last row (NaN target) for training
df = df.dropna()

# Features
features = ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
            'precipitation', 'pressure_msl', 'cloud_cover', 'visibility',
            'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m',
            'hour', 'day', 'month', 'weekday']

X = df[features]
y = df['temperature_next_hour']  # Target: next hour temperature

# -------------------- SPLIT DATA --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------- TRAIN MODEL --------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -------------------- EVALUATE MODEL --------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# -------------------- SAVE MODEL --------------------
joblib.dump(model, "hourly_weather_model.pkl")
print("Model saved as 'hourly_weather_model.pkl'")
