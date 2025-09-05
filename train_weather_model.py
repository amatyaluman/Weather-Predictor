import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from prophet import Prophet
import joblib

# -------------------- LOAD DATA --------------------
df = pd.read_csv("open-meteo-27.75N85.50E1293mNew.csv")
df['time'] = pd.to_datetime(df['time'])

# -------------------- FEATURE ENGINEERING (RF) --------------------
df['hour'] = df['time'].dt.hour
df['day'] = df['time'].dt.day
df['month'] = df['time'].dt.month
df['weekday'] = df['time'].dt.weekday

df['temperature_next_hour'] = df['temperature_2m'].shift(-1)
df = df.dropna()

features = [
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
    'precipitation', 'pressure_msl', 'cloud_cover', 'visibility',
    'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m',
    'hour', 'day', 'month', 'weekday'
]

X = df[features]
y = df['temperature_next_hour']

# -------------------- RANDOM FOREST --------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
print("Random Forest → MSE:", mean_squared_error(y_test, y_pred))
print("Random Forest → R²:", r2_score(y_test, y_pred))

joblib.dump(rf_model, "hourly_weather_model.pkl")
print("Saved Random Forest model → hourly_weather_model.pkl")

# -------------------- PROPHET --------------------
df_prophet = df.rename(columns={'time': 'ds'})
features_to_forecast = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'precipitation']

all_prophet_models = {}
for feature in features_to_forecast:
    temp_df = df_prophet[['ds', feature]].rename(columns={feature: 'y'})
    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(temp_df)
    all_prophet_models[feature] = model
    print(f"Trained Prophet model for {feature}")

joblib.dump(all_prophet_models, "prophet_kathmandu_all_models.pkl")
print("Saved Prophet models → prophet_kathmandu_all_models.pkl")
