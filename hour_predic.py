import pandas as pd
import streamlit as st
import joblib
from datetime import datetime
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Weather Dashboard",
    layout="wide",
    page_icon="weather_icon.png",
    initial_sidebar_state="expanded"
)

# -------------------- CSS --------------------
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    :root {
        --primary: #2b313e;
        --secondary: #3b4252;
        --accent: #81a1c1;
        --text: #e5e9f0;
        --highlight: #88c0d0;
    }
    body, .stApp {
        background-color: var(--primary);
        color: var(--text);
        font-family: 'Segoe UI', sans-serif;
        line-height: 1.6;
    }
    .card, .metric-card {
        padding: 20px;
        border-radius: 10px;
        background-color: var(--secondary);
        margin-bottom: 20px;
        border-left: 4px solid var(--accent);
        min-width: 150px;
        text-align: center;
    }
    .weather-icon {
        font-size: 24px;
        margin-bottom: 5px;
        color: var(--accent);
    }
    .highlight {
        background: linear-gradient(90deg, #5e81ac, #81a1c1);
        padding: 2px 8px;
        border-radius: 4px;
        color: white;
        font-weight: bold;
    }
    .hourly-forecast {
        display: flex;
        overflow-x: auto;
        gap: 10px;
        padding-bottom: 10px;
    }
    .stDateInput, .stTextInput input, .stNumberInput input {
        background-color: var(--secondary) !important;
        color: var(--text) !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("<h2><i class='fas fa-compass'></i> Navigation</h2>", unsafe_allow_html=True)
    page = st.radio(
        "Go to:",
        options=["Prediction", "Trends"],
        format_func=lambda x: f"{x}"
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
        <h3><i class="fas fa-info-circle"></i> About</h3>
        <p>This dashboard provides forecasts and historical trends for <b>Kathmandu, Nepal</b>.<br>
        Data is sourced from <b>Open-Meteo API</b> and analyzed using <b>machine learning models</b>.</p>
    """, unsafe_allow_html=True)

# -------------------- Weather mapping --------------------
weather_mapping = {
    0: ("Clear sky", "fa-sun"),
    1: ("Mainly clear", "fa-cloud-sun"),
    2: ("Partly cloudy", "fa-cloud"),
    3: ("Overcast", "fa-cloud"),
    45: ("Fog", "fa-smog"),
    48: ("Rime fog", "fa-smog"),
    51: ("Light drizzle", "fa-cloud-rain"),
    53: ("Moderate drizzle", "fa-cloud-rain"),
    55: ("Heavy drizzle", "fa-cloud-showers-heavy"),
    61: ("Light rain", "fa-cloud-rain"),
    63: ("Moderate rain", "fa-cloud-showers-heavy"),
    65: ("Heavy rain", "fa-cloud-showers-heavy"),
    71: ("Light snow", "fa-snowflake"),
    73: ("Moderate snow", "fa-snowflake"),
    75: ("Heavy snow", "fa-snowflake"),
    80: ("Rain showers", "fa-cloud-rain"),
    81: ("Violent rain showers", "fa-poo-storm"),
    95: ("Thunderstorm", "fa-bolt")
}

# -------------------- Load model & data --------------------
@st.cache_resource
def load_model():
    return joblib.load("weather_model.pkl")

@st.cache_data
def load_historical_data():
    df = pd.read_csv("open-meteo-27.73N85.25E1293m.csv")
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month
    model = load_model()
    for feature in model.feature_names_in_:
        if feature not in df.columns:
            df[feature] = 0
    return df

model = load_model()
historical_data = load_historical_data()

# -------------------- Forecast function --------------------
def generate_forecast_hourly(requested_date):
    similar_data = historical_data[
        (historical_data['month'] == requested_date.month) &
        (historical_data['day_of_week'] == requested_date.weekday())
    ]
    if similar_data.empty:
        similar_data = historical_data

    predictions = []
    for hour in range(24):
        hour_data = similar_data[similar_data['hour'] == hour]
        if hour_data.empty:
            continue
        avg_row = hour_data.mean(numeric_only=True)
        avg_row['hour'] = hour
        avg_row['day_of_week'] = requested_date.weekday()
        avg_row['month'] = requested_date.month
        input_df = pd.DataFrame([avg_row])[model.feature_names_in_]
        predicted_temp = model.predict(input_df)[0]
        weather_code = int(hour_data['weather_code'].mode()[0])
        predictions.append({
            'hour': hour,
            'predicted_temp': round(predicted_temp, 1),
            'weather_code': weather_code
        })
    return predictions

# -------------------- Pages --------------------
if page == "Prediction":
    st.markdown("<h1 style='color: var(--accent);'><i class='fas fa-cloud-sun weather-icon'></i> Kathmandu Hourly Weather Forecast</h1>", unsafe_allow_html=True)

    selected_date = st.date_input("Select a date", value=datetime.now().date(), min_value=historical_data['date'].min())

    if selected_date >= datetime.now().date():
        with st.spinner("Generating forecast..."):
            forecast = generate_forecast_hourly(selected_date)

        if forecast:
            st.markdown(f"<div class='card'><h2>Forecast for {selected_date.strftime('%A, %B %d, %Y')}</h2></div>", unsafe_allow_html=True)

            # Hourly cards
            st.markdown('<div class="hourly-forecast">', unsafe_allow_html=True)
            for f in forecast:
                hour = f"{f['hour']:02d}:00"
                condition, icon = weather_mapping.get(f['weather_code'], ("Unknown", "fa-question"))
                st.markdown(f"""
                <div class='metric-card'>
                    <i class='fas {icon} weather-icon'></i>
                    <h4>{hour}</h4>
                    <p>Temp: <span class='highlight'>{f['predicted_temp']}°C</span></p>
                    <p>Condition: {condition}</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Plotly line chart
            df_chart = pd.DataFrame(forecast)
            fig = px.line(df_chart, x='hour', y='predicted_temp', markers=True,
                          labels={'hour':'Hour', 'predicted_temp':'Temperature (°C)'},
                          title=f"Temperature Forecast for {selected_date.strftime('%A, %b %d')}")
            fig.update_layout(plot_bgcolor="#2b313e", paper_bgcolor="#2b313e", font_color="#e5e9f0")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Not enough historical data for prediction.")
    else:
        st.warning("Please select today or a future date.")

elif page == "Trends":
    st.markdown("<h1 style='color: var(--accent);'><i class='fas fa-chart-bar weather-icon'></i> Explore Historical Weather Trends</h1>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Temperature vs Humidity", "Wind Speed Distribution"])

    with tab1:
        st.markdown("<div class='card'><h3>Temperature vs Humidity</h3></div>", unsafe_allow_html=True)
        scatter_df = historical_data[['temperature_2m', 'relative_humidity_2m']].dropna()
        fig = px.scatter(scatter_df, x='relative_humidity_2m', y='temperature_2m',
                         color='temperature_2m', color_continuous_scale='reds',
                         labels={'relative_humidity_2m':'Humidity (%)','temperature_2m':'Temp (°C)'})
        fig.update_layout(plot_bgcolor="#2b313e", paper_bgcolor="#2b313e", font_color="#e5e9f0")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("<div class='card'><h3>Wind Speed Distribution</h3></div>", unsafe_allow_html=True)
        wind_df = historical_data[['wind_speed_10m (km/h)']].dropna()
        fig_wind = px.histogram(wind_df, x='wind_speed_10m (km/h)', nbins=30, color_discrete_sequence=['#81a1c1'])
        fig_wind.update_layout(plot_bgcolor="#2b313e", paper_bgcolor="#2b313e", font_color="#e5e9f0")
        st.plotly_chart(fig_wind, use_container_width=True)
