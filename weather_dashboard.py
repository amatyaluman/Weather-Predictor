import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta
import plotly.express as px

# -------------------- LOAD MODEL --------------------
model = joblib.load("hourly_weather_model.pkl")

st.set_page_config(page_title="Kathmandu Weather Forecast", layout="wide")
st.title("Kathmandu Hourly Weather Forecast 🌤️")

# -------------------- GET CURRENT WEATHER FROM OPEN-METEO --------------------
latitude = 27.7017
longitude = 85.3206

api_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,pressure_msl,cloudcover,visibility,windspeed_10m,winddirection_10m,windgusts_10m"

response = requests.get(api_url)
data = response.json()

if "current_weather" in data:
    current = data["current_weather"]

    temperature = current["temperature"]
    humidity = current.get("humidity", 60)
    dew_point = current.get("dew_point", temperature - 5)
    wind_speed = current["windspeed"]
    wind_direction = current["winddirection"]
    wind_gusts = current.get("windgusts", wind_speed)
    pressure = current.get("pressure_msl", 1013)
    cloud_cover = current.get("cloudcover", 50)
    visibility = current.get("visibility", 10000)
    precipitation = current.get("precipitation", 0)
    
    st.subheader("Current Weather in Kathmandu")
    st.write(f"Temperature: {temperature} °C")
    st.write(f"Humidity: {humidity} %")
    st.write(f"Wind Speed: {wind_speed} km/h")
    st.write(f"Pressure: {pressure} hPa")
else:
    st.error("Failed to fetch current weather data.")
    st.stop()

# -------------------- PREDICT NEXT HOURS --------------------
next_hours = st.slider("Forecast Next Hours", 1, 12, 5)

forecast = []
current_time = datetime.now()

for hour in range(next_hours):
    hour_feature = (current_time + timedelta(hours=hour)).hour
    day = (current_time + timedelta(hours=hour)).day
    month = (current_time + timedelta(hours=hour)).month
    weekday = (current_time + timedelta(hours=hour)).weekday()
    
    features = np.array([[temperature, humidity, dew_point, precipitation,
                          pressure, cloud_cover, visibility, wind_speed,
                          wind_direction, wind_gusts, hour_feature, day, month, weekday]])
    
    next_temp = model.predict(features)[0]
    forecast.append({'time': current_time + timedelta(hours=hour+1),
                     'predicted_temperature': next_temp})
    
    # update temperature for next prediction
    temperature = next_temp

forecast_df = pd.DataFrame(forecast)

# -------------------- DISPLAY RESULTS --------------------
st.subheader("Predicted Hourly Temperature")
st.dataframe(forecast_df)

fig = px.line(forecast_df, x='time', y='predicted_temperature',
              labels={'time': 'Time', 'predicted_temperature': 'Temperature (°C)'},
              title="Next Hourly Temperature Forecast for Kathmandu")
st.plotly_chart(fig, use_container_width=True)
