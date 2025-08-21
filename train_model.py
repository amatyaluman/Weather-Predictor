import pandas as pd
from prophet import Prophet
import joblib

# Load CSV
df = pd.read_csv("open-meteo-27.75N85.50E1293mNew.csv")
df['ds'] = pd.to_datetime(df['time'])

# Columns to forecast
features = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'precipitation']

# Dictionary to hold models
all_models = {}

# Train models
for feature in features:
    df['y'] = df[feature]
    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(df[['ds','y']])
    all_models[feature] = model
    print(f"Trained model for {feature}")

# Save all models in one file
joblib.dump(all_models, "prophet_kathmandu_all_models.pkl")
print("All models saved in prophet_kathmandu_all_models.pkl")
