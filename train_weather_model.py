# train_weather_model_optimized.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

df = pd.read_csv("open-meteo-27.73N85.25E1293m.csv")

# Feature engineering
df['time'] = pd.to_datetime(df['time'])
df['hour'] = df['time'].dt.hour
df['month'] = df['time'].dt.month
df['day_of_week'] = df['time'].dt.weekday

df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
df['month_cos'] = np.cos(2 * np.pi * df['month']/12)

bins = [0, 1, 3, 50, 70, 100]
labels = ['clear', 'cloudy', 'fog', 'rain', 'storm']
df['weather_type'] = pd.cut(df['weather_code'], bins=bins, labels=labels)
df = pd.get_dummies(df, columns=['weather_type'])

# Features and target
feature_cols = [
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'hour', 'month', 'day_of_week',
    'relative_humidity_2m', 'wind_speed_10m (km/h)', 'wind_direction_100m'
]
weather_dummies = [col for col in df.columns if col.startswith('weather_type_')]
feature_cols.extend(weather_dummies)

X = df[feature_cols]
y = df[['temperature_2m']]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optimized Random Forest
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.2f}")

# Save optimized model with compression
joblib.dump(model, "optimized_weather_model.pkl", compress=3)
print("Optimized model saved as optimized_weather_model.pkl")
##bb