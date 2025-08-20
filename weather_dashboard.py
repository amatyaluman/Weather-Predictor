import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

st.set_page_config(
    page_title="Weather Forecast Dashboard",
    layout="wide",
    page_icon="☀️",
    initial_sidebar_state="expanded"
)

def load_css():
    st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined&display=swap" rel="stylesheet" />
        <style>
        body {
          font-family: "Roboto", "Helvetica", sans-serif;
          color: #f1f5f9;
          background-color: #0f172a;
        }
        .current-weather {
          background-color: #1e293b;
          padding: 20px;
          border-radius: 16px;
          margin-bottom: 20px;
        }
        .disclaimer {
          background-color: #1e293b;
          color: #f59e0b;
          padding: 16px;
          border-radius: 10px;
          margin: 16px 0;
          border-left: 4px solid #f59e0b;
        }
        .material-symbols-outlined {
          font-family: 'Material Symbols Outlined';
          font-style: normal;
          font-weight: normal;
          font-size: 48px;
          line-height: 1;
          display: inline-block;
          vertical-align: middle;
          -webkit-font-smoothing: antialiased;
        }
        </style>
    """, unsafe_allow_html=True)

def get_weather_icon_and_text(code):
    mapping = {
        0: ("wb_sunny", "Clear", "#facc15"),
        1: ("wb_sunny", "Mostly Sunny", "#facc15"),
        2: ("partly_cloudy_day", "Partly Cloudy", "#94a3b8"),
        3: ("cloud", "Cloudy", "#64748b"),
        61: ("rainy", "Rain", "#0284c7"),
        95: ("thunderstorm", "Thunderstorm", "#dc2626")
    }
    return mapping.get(int(code), ("help_outline", "Unknown", "#94a3b8"))

def main():
    load_css()
    st.title("🌤️ Weather Forecast Dashboard")

    st.markdown('<div class="current-weather"><h2>Current Weather</h2></div>', unsafe_allow_html=True)

    # Example code (replace with your live data later)
    current_code = 2
    icon, text, color = get_weather_icon_and_text(current_code)

    st.markdown(
        f"""
        <div style="text-align:center; font-size:1.2rem;">
            <span class="material-symbols-outlined" style="color:{color};">{icon}</span><br>
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="disclaimer">⚠️ Data may vary from official weather channels.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
