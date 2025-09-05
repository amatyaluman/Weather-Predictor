import joblib
import pandas as pd

# -------------------- LOAD MODELS --------------------
rf_model = joblib.load("hourly_weather_model.pkl")
all_prophet_models = joblib.load("prophet_kathmandu_all_models.pkl")

# -------------------- RANDOM FOREST (NEXT HOUR) --------------------
sample = pd.DataFrame([{
    'temperature_2m': 26,
    'relative_humidity_2m': 70,
    'dew_point_2m': 19,
    'precipitation': 0.5,
    'pressure_msl': 1012,
    'cloud_cover': 60,
    'visibility': 12000,
    'wind_speed_10m': 12,
    'wind_direction_10m': 200,
    'wind_gusts_10m': 22,
    'hour': 15,
    'day': 5,
    'month': 9,
    'weekday': 4
}])

next_hour_temp = rf_model.predict(sample)[0]
print("Predicted Next Hour Temperature:", round(next_hour_temp, 2), "°C")

# -------------------- PROPHET (MULTI-HOUR FORECAST) --------------------
model = all_prophet_models['temperature_2m']  # choose variable
future = model.make_future_dataframe(periods=24, freq="H")  # next 24 hours
forecast = model.predict(future)

print("\nNext 5 forecasted temperature values:")
print(forecast[['ds', 'yhat']].tail(5))
