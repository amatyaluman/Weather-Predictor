import pandas as pd
import streamlit as st
import joblib
from datetime import datetime
import numpy as np
import altair as alt

st.set_page_config(
    page_title="Weather Dashboard",
    layout="wide",
    page_icon="weather_icon.png",
    initial_sidebar_state="expanded"
)

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
    }

    .weather-icon {
        font-size: 24px;
        margin-right: 10px;
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
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 15px;
    }

    div[data-baseweb="radio"] > div {
        background-color: transparent !important;
        box-shadow: none !important;
        padding-left: 0 !important;
    }

    .stDateInput, .stTextInput input, .stNumberInput input {
        background-color: var(--secondary) !important;
        color: var(--text) !important;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
        <h2><i class='fas fa-compass'></i> Navigation</h2>
    """, unsafe_allow_html=True)

    selected = st.session_state.get("nav_page", "Prediction")

    if st.button(" Prediction", key="pred_btn"):
        st.session_state["nav_page"] = "Prediction"
    if st.button("Trends", key="trend_btn"):
        st.session_state["nav_page"] = "Trends"

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
        <h3><i class="fas fa-info-circle"></i> About</h3>
        <p style="font-size: 0.9rem;">
        This dashboard provides weather forecasts and historical trends for <b>Kathmandu, Nepal</b>.<br>
        Data is sourced from <b>Open-Meteo API</b> and analyzed using <b>machine learning models</b>.
        </p>
    """, unsafe_allow_html=True)

page = st.session_state.get("nav_page", "Prediction")

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

alt.themes.register('dark', lambda: {
    'config': {
        'background': '#2b313e',
        'title': {'color': '#e5e9f0'},
        'axis': {
            'labelColor': '#e5e9f0',
            'titleColor': '#e5e9f0',
            'gridColor': '#3b4252',
            'domainColor': '#e5e9f0'
        },
        'legend': {
            'labelColor': '#e5e9f0',
            'titleColor': '#e5e9f0'
        }
    }
})
alt.themes.enable('dark')

if page == "Prediction":
    st.markdown("<h1 style='color: var(--accent);'><i class='fas fa-cloud-sun weather-icon'></i> Kathmandu Hourly Weather Forecast</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        selected_date = st.date_input("Select a date", value=datetime.now().date(), min_value=historical_data['date'].min())

#    with col2:
      #  st.markdown("""
        #<div class='card'>
         #   <h3><i class='fas fa-info-circle weather-icon'></i> About Predictions</h3>
          #  <p>Forecasts are generated using ML models trained on historical weather.</p>
        #</div>
        #""", unsafe_allow_html=True)

    if selected_date >= datetime.now().date():
        with st.spinner("<i class='fas fa-cog fa-spin'></i> Generating forecast..."):
            forecast = generate_forecast_hourly(selected_date)

        if forecast:
            st.markdown(f"""
            <div class='card'>
                <h2><i class='fas fa-calendar-day weather-icon'></i> Forecast for {selected_date.strftime('%A, %B %d, %Y')}</h2>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="hourly-forecast">', unsafe_allow_html=True)
            for f in forecast:
                hour = f"{f['hour']:02d}:00"
                condition, icon = weather_mapping.get(f['weather_code'], ("Unknown", "fa-question"))
                st.markdown(f"""
                <div class='metric-card'>
                    <h4><i class='fas {icon} weather-icon'></i> {hour}</h4>
                    <p><i class='fas fa-temperature-high weather-icon'></i> Temp: <span class='highlight'>{f['predicted_temp']}°C</span></p>
                    <p><i class='fas fa-cloud weather-icon'></i> Condition: {condition}</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            df_chart = pd.DataFrame(forecast)
            chart = alt.Chart(df_chart).mark_line(point=alt.OverlayMarkDef(filled=True, fill='#81a1c1'), strokeWidth=2).encode(
                x=alt.X('hour:O', title='Hour'),
                y=alt.Y('predicted_temp:Q', title='Temperature (°C)'),
                tooltip=['hour', 'predicted_temp']
            ).properties(height=400)
            st.altair_chart(chart, use_container_width=True)

        else:
            st.error("Not enough historical data for prediction.")
    else:
        st.warning("Please select today or a future date.")

elif page == "Trends":
    st.markdown("<h1 style='color: var(--accent);'><i class='fas fa-chart-bar weather-icon'></i> Explore Historical Weather Trends</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <h3><i class='fas fa-temperature-high weather-icon'></i> Temperature vs Humidity</h3>
    </div>
    """, unsafe_allow_html=True)

    scatter_df = historical_data[['temperature_2m', 'relative_humidity_2m']].dropna()
    scatter = alt.Chart(scatter_df).mark_circle(size=60, opacity=0.7).encode(
        x='relative_humidity_2m:Q',
        y='temperature_2m:Q',
        color=alt.Color('temperature_2m:Q', scale=alt.Scale(scheme='redyellowblue')),
        tooltip=['relative_humidity_2m', 'temperature_2m']
    ).interactive()
    st.altair_chart(scatter, use_container_width=True)

    st.markdown("""
    <div class='card'>
        <h3><i class='fas fa-wind weather-icon'></i> Wind Speed Distribution</h3>
    </div>
    """, unsafe_allow_html=True)

    wind_df = historical_data[['wind_speed_10m (km/h)']].dropna()
    hist = alt.Chart(wind_df).mark_bar(color='#81a1c1').encode(
        x=alt.X('wind_speed_10m (km/h):Q', bin=alt.Bin(maxbins=30)),
        y='count()'
    ).properties(height=300)
    st.altair_chart(hist, use_container_width=True)
