import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
from math import ceil
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Kathmandu Weather Forecast",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        border-bottom: 2px solid #64B5F6;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #E3F2FD;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        text-align: center;
    }
    .hour-label {
        font-weight: bold;
        color: #1565C0;
        margin-bottom: 0.5rem;
    }
    .stDateInput > div > div > input {
        background-color: #E3F2FD;
    }
    .css-1d391kg, .css-12oz5g7 {
        padding-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Grid View", "Charts", "Summary"])

# Date input with improved defaults
min_date = datetime.now().date()
max_date = min_date + timedelta(days=7)
forecast_date = st.sidebar.date_input(
    "Select Forecast Date", 
    value=min_date,
    min_value=min_date,
    max_value=max_date,
    help="Select a date within the next 7 days"
)

# Generate 24 hourly timestamps
start = datetime.combine(forecast_date, datetime.min.time())
future_dates = pd.date_range(start=start, periods=24, freq='H')
future_df = pd.DataFrame({'ds': future_dates})

# Load all models
@st.cache_resource
def load_models():
    return joblib.load("prophet_kathmandu_all_models.pkl")

all_models = load_models()

# Predict each feature
forecast_results = pd.DataFrame({'time': future_dates})
for feature, model in all_models.items():
    forecast = model.predict(future_df)
    forecast_results[feature] = forecast['yhat']

# Add time-based labels
forecast_results['hour'] = forecast_results['time'].dt.hour
forecast_results['time_of_day'] = forecast_results['hour'].apply(
    lambda x: 'Morning' if 5 <= x < 12 else 'Afternoon' if 12 <= x < 17 else 'Evening' if 17 <= x < 21 else 'Night'
)

# -------------------- GRID VIEW --------------------
if page == "Grid View":
    st.markdown('<h1 class="main-header">Kathmandu Hourly Weather Forecast 🌤️</h1>', unsafe_allow_html=True)
    st.markdown(f'<h2 class="sub-header">Hourly Forecast for {forecast_date}</h2>', unsafe_allow_html=True)
    
    # Day summary metrics
    temp_min = forecast_results['temperature_2m'].min()
    temp_max = forecast_results['temperature_2m'].max()
    avg_humidity = forecast_results['relative_humidity_2m'].mean()
    total_precip = forecast_results['precipitation'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Min Temperature", f"{temp_min:.1f}°C")
    col2.metric("Max Temperature", f"{temp_max:.1f}°C")
    col3.metric("Avg Humidity", f"{avg_humidity:.1f}%")
    col4.metric("Total Precipitation", f"{total_precip:.1f}mm")
    
    st.markdown("---")
    
    # Time of day selector
    time_filter = st.selectbox("Filter by Time of Day", ["All Day", "Morning", "Afternoon", "Evening", "Night"])
    
    if time_filter != "All Day":
        filtered_data = forecast_results[forecast_results['time_of_day'] == time_filter]
    else:
        filtered_data = forecast_results
    
    # Display hourly forecasts in a grid
    cols_per_row = 4
    num_rows = ceil(len(filtered_data) / cols_per_row)
    
    for i in range(num_rows):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i * cols_per_row + j
            if idx < len(filtered_data):
                hour_data = filtered_data.iloc[idx]
                
                # Determine appropriate icons based on conditions
                temp_icon = "❄️" if hour_data['temperature_2m'] < 10 else "🌡️" if hour_data['temperature_2m'] < 20 else "🔥"
                humidity_icon = "💧"
                wind_icon = "💨"
                precip_icon = "🌧️" if hour_data['precipitation'] > 0 else "☂️"
                
                with col:
                    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="hour-label">{hour_data["time"].strftime("%H:%M")} ({hour_data["time_of_day"]})</div>', unsafe_allow_html=True)
                    st.metric(f"{temp_icon} Temp (°C)", f"{hour_data['temperature_2m']:.1f}")
                    st.metric(f"{humidity_icon} Humidity (%)", f"{hour_data['relative_humidity_2m']:.0f}")
                    st.metric(f"{wind_icon} Wind (km/h)", f"{hour_data['wind_speed_10m']:.1f}")
                    st.metric(f"{precip_icon} Precip (mm)", f"{hour_data['precipitation']:.1f}")
                    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- CHARTS VIEW --------------------
elif page == "Charts":
    st.markdown('<h1 class="main-header">Kathmandu Hourly Weather Charts 📊</h1>', unsafe_allow_html=True)
    st.markdown(f'<h2 class="sub-header">Forecast Charts for {forecast_date}</h2>', unsafe_allow_html=True)
    
    # Create a dropdown to select which chart to view
    chart_option = st.selectbox(
        "Select Chart Type",
        ["Temperature", "Humidity", "Wind Speed", "Precipitation", "All Charts", "Combined View"]
    )
    
    if chart_option == "All Charts":
        for feature in all_models.keys():
            fig = px.line(forecast_results, x='time', y=feature,
                          labels={'time': 'Time', feature: feature.replace('_', ' ').title()},
                          title=f"{feature.replace('_', ' ').title()} Forecast")
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title=feature.replace('_', ' ').title(),
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif chart_option == "Combined View":
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Temperature (°C)", "Humidity (%)", "Wind Speed (km/h)", "Precipitation (mm)")
        )
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=forecast_results['time'], y=forecast_results['temperature_2m'], name="Temperature"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=forecast_results['time'], y=forecast_results['relative_humidity_2m'], name="Humidity"),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=forecast_results['time'], y=forecast_results['wind_speed_10m'], name="Wind Speed"),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=forecast_results['time'], y=forecast_results['precipitation'], name="Precipitation"),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(height=700, showlegend=False, title_text="Combined Weather Forecast")
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=2)
        fig.update_yaxes(title_text="°C", row=1, col=1)
        fig.update_yaxes(title_text="%", row=1, col=2)
        fig.update_yaxes(title_text="km/h", row=2, col=1)
        fig.update_yaxes(title_text="mm", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        feature_map = {
            "Temperature": "temperature_2m",
            "Humidity": "relative_humidity_2m",
            "Wind Speed": "wind_speed_10m",
            "Precipitation": "precipitation"
        }
        
        feature = feature_map[chart_option]
        fig = px.line(forecast_results, x='time', y=feature,
                      labels={'time': 'Time', feature: chart_option},
                      title=f"{chart_option} Forecast")
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title=chart_option,
            hovermode="x unified"
        )
        
        # Add a horizontal line for average if not precipitation
        if chart_option != "Precipitation":
            avg_value = forecast_results[feature].mean()
            fig.add_hline(y=avg_value, line_dash="dash", line_color="red", 
                         annotation_text=f"Average: {avg_value:.1f}")
        
        st.plotly_chart(fig, use_container_width=True)

# -------------------- SUMMARY VIEW --------------------
elif page == "Summary":
    st.markdown('<h1 class="main-header">Kathmandu Weather Summary 📋</h1>', unsafe_allow_html=True)
    st.markdown(f'<h2 class="sub-header">Daily Summary for {forecast_date}</h2>', unsafe_allow_html=True)
    
    # Calculate summary statistics
    temp_stats = {
        'Min': forecast_results['temperature_2m'].min(),
        'Max': forecast_results['temperature_2m'].max(),
        'Average': forecast_results['temperature_2m'].mean()
    }
    
    humidity_stats = {
        'Min': forecast_results['relative_humidity_2m'].min(),
        'Max': forecast_results['relative_humidity_2m'].max(),
        'Average': forecast_results['relative_humidity_2m'].mean()
    }
    
    wind_stats = {
        'Min': forecast_results['wind_speed_10m'].min(),
        'Max': forecast_results['wind_speed_10m'].max(),
        'Average': forecast_results['wind_speed_10m'].mean()
    }
    
    precip_total = forecast_results['precipitation'].sum()
    precip_hours = len(forecast_results[forecast_results['precipitation'] > 0])
    
    # Display summary in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Temperature (°C)")
        for stat, value in temp_stats.items():
            st.metric(stat, f"{value:.1f}")
    
    with col2:
        st.markdown("### Humidity (%)")
        for stat, value in humidity_stats.items():
            st.metric(stat, f"{value:.1f}")
    
    with col3:
        st.markdown("### Wind Speed (km/h)")
        for stat, value in wind_stats.items():
            st.metric(stat, f"{value:.1f}")
    
    st.markdown("---")
    
    # Precipitation summary
    col4, col5 = st.columns(2)
    
    with col4:
        st.markdown("### Precipitation")
        st.metric("Total Precipitation", f"{precip_total:.1f} mm")
        st.metric("Hours with Precipitation", f"{precip_hours}")
    
    with col5:
        st.markdown("### Time Distribution")
        time_counts = forecast_results['time_of_day'].value_counts()
        fig = px.pie(values=time_counts.values, names=time_counts.index, 
                     title="Time of Day Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Best time of day recommendations
    st.markdown("### Recommended Times")
    
    # Find the most comfortable time (moderate temp and humidity)
    comfort_scores = []
    for idx, row in forecast_results.iterrows():
        # Simple comfort score (lower is better)
        temp_diff = abs(row['temperature_2m'] - 22)  # Ideal temperature around 22°C
        humidity_diff = abs(row['relative_humidity_2m'] - 50)  # Ideal humidity around 50%
        comfort_score = temp_diff + humidity_diff
        comfort_scores.append(comfort_score)
    
    forecast_results['comfort_score'] = comfort_scores
    best_times = forecast_results.nsmallest(3, 'comfort_score')
    
    st.write("Most comfortable times of day (pleasant temperature and humidity):")
    for idx, time in best_times.iterrows():
        st.write(f"- {time['time'].strftime('%H:%M')} ({time['time_of_day']}): "
                f"{time['temperature_2m']:.1f}°C, {time['relative_humidity_2m']:.0f}% humidity")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    """
    This app provides weather forecasts for Kathmandu using Prophet models.
    Data is updated regularly for accurate predictions.
    """
)