#train_weather_model
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import joblib
import holidays
from prophet import Prophet

print("Loading historical data...")
df = pd.read_csv("open-meteo-27.75N85.50E1293mNew.csv")
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

# Feature Engineering
df['hour'] = df['time'].dt.hour
df['day'] = df['time'].dt.day
df['month'] = df['time'].dt.month
df['weekday'] = df['time'].dt.weekday
df['day_of_year'] = df['time'].dt.dayofyear
df['is_weekend'] = df['weekday'].isin([5,6]).astype(int)

# Nepal Holidays
np_holidays = holidays.Nepal()
df['is_holiday'] = df['time'].apply(lambda x: x in np_holidays).astype(int)

# Lag Features
features_base = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'precipitation']
for col in features_base:
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)

# Rolling statistics
for window in [3, 6, 12, 24]:
    df[f'temp_rolling_mean_{window}'] = df['temperature_2m'].rolling(window=window).mean()
    df[f'temp_rolling_std_{window}'] = df['temperature_2m'].rolling(window=window).std()
    df[f'humidity_rolling_mean_{window}'] = df['relative_humidity_2m'].rolling(window=window).mean()

# Create weather type labels based on conditions
def categorize_weather(row):
    """
    Categorize weather based on temperature, humidity, precipitation, and wind
    """
    temp = row['temperature_2m']
    humidity = row['relative_humidity_2m']
    precipitation = row['precipitation']
    wind_speed = row['wind_speed_10m']
    
    if precipitation > 5.0:
        return 'Heavy Rain'
    elif precipitation > 0.5:
        return 'Rain'
    elif precipitation > 0.1:
        return 'Light Rain'
    elif humidity > 90:
        return 'Foggy'
    elif humidity > 80:
        return 'Humid'
    elif wind_speed > 25:
        return 'Windy'
    elif wind_speed > 15:
        return 'Breezy'
    elif temp > 30:
        return 'Hot'
    elif temp > 25:
        return 'Warm'
    elif temp > 15:
        return 'Mild'
    elif temp > 5:
        return 'Cool'
    elif temp <= 0:
        return 'Freezing'
    else:
        return 'Clear'

# Apply weather categorization
df['weather_type'] = df.apply(categorize_weather, axis=1)

# Alternative: Use weather_code if available in your data
weather_code_mapping = {
    0: 'Clear', 1: 'Mainly Clear', 2: 'Partly Cloudy', 3: 'Overcast',
    45: 'Foggy', 48: 'Foggy', 51: 'Light Drizzle', 53: 'Drizzle', 55: 'Heavy Drizzle',
    56: 'Light Freezing Drizzle', 57: 'Freezing Drizzle', 61: 'Light Rain', 
    63: 'Rain', 65: 'Heavy Rain', 66: 'Light Freezing Rain', 67: 'Freezing Rain',
    71: 'Light Snow', 73: 'Snow', 75: 'Heavy Snow', 77: 'Snow Grains',
    80: 'Light Showers', 81: 'Showers', 82: 'Heavy Showers', 85: 'Light Snow Showers',
    86: 'Snow Showers', 95: 'Thunderstorm', 96: 'Thunderstorm with Hail', 
    99: 'Thunderstorm with Heavy Hail'
}

# If weather_code column exists, use it for more accurate weather types
if 'weather_code' in df.columns:
    df['weather_type'] = df['weather_code'].map(weather_code_mapping).fillna('Clear')

df.dropna(inplace=True)

# Features for the model
feature_cols = ['hour', 'day', 'month', 'weekday', 'day_of_year', 'is_weekend', 'is_holiday']
for col in features_base:
    for lag in [1, 2, 3, 6, 12, 24]:
        feature_cols.append(f'{col}_lag_{lag}')
for col in df.columns:
    if 'rolling' in col:
        feature_cols.append(col)

# Train Random Forest Models for numerical predictions
rf_models = {}
for target in features_base:
    print(f"\nTraining Random Forest model for {target}...")
    X = df[feature_cols]
    y = df[target]

    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    rmses = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        rmses.append(rmse)

    print(f"{target} → CV RMSE: {np.mean(rmses):.3f} ± {np.std(rmses):.3f}")

    # Train final model on all data
    final_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    final_model.fit(X, y)
    rf_models[target] = final_model

# Train Random Forest Classifier for weather type
print("\nTraining Random Forest model for weather type...")
X_weather = df[feature_cols]
y_weather = df['weather_type']

# TimeSeriesSplit for weather type
tscv_weather = TimeSeriesSplit(n_splits=5)
accuracies = []

for train_index, test_index in tscv_weather.split(X_weather):
    X_train, X_test = X_weather.iloc[train_index], X_weather.iloc[test_index]
    y_train, y_test = y_weather.iloc[train_index], y_weather.iloc[test_index]

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

print(f"Weather Type → CV Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")

# Train final weather type model on all data
weather_type_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
weather_type_model.fit(X_weather, y_weather)
rf_models['weather_type'] = weather_type_model

# Save Random Forest models
joblib.dump(rf_models, "hourly_weather_model.pkl")
print("\nAll Random Forest models saved as hourly_weather_model.pkl")

# Train Prophet models for daily forecasts
print("\nTraining Prophet models for daily forecasts...")
prophet_models = {}

# Prepare data for Prophet (daily aggregation)
df_daily = df.copy()
df_daily['date'] = df_daily['time'].dt.date
daily_agg = df_daily.groupby('date').agg({
    'temperature_2m': 'mean',
    'relative_humidity_2m': 'mean', 
    'wind_speed_10m': 'mean',
    'precipitation': 'sum'
}).reset_index()

for target in features_base:
    print(f"Training Prophet model for {target}...")
    
    # Prepare data in Prophet format
    prophet_df = daily_agg[['date', target]].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df = prophet_df.dropna()
    
    # Create and fit Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    
    model.fit(prophet_df)
    prophet_models[target] = model

# Save Prophet models
joblib.dump(prophet_models, "prophet_models.pkl")
print("All Prophet models saved as prophet_models.pkl")

print("\nTraining completed successfully!")
print("Available models:")
print("- Temperature prediction")
print("- Humidity prediction") 
print("- Wind speed prediction")
print("- Precipitation prediction")
print("- Weather type classification")