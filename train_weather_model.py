# train_weather_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
import joblib
import holidays

# -------------------- LOAD HISTORICAL DATA --------------------
df = pd.read_csv("open-meteo-27.75N85.50E1293mNew.csv")
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

# -------------------- FEATURE ENGINEERING (RF) --------------------
df['hour'] = df['time'].dt.hour
df['day'] = df['time'].dt.day
df['month'] = df['time'].dt.month
df['weekday'] = df['time'].dt.weekday
df['day_of_year'] = df['time'].dt.dayofyear
df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

# Add holiday information for Nepal
np_holidays = holidays.Nepal()
df['is_holiday'] = df['time'].apply(lambda x: x in np_holidays).astype(int)

# Lag features
for lag in [1, 2, 3, 6, 12, 24]:
    df[f'temp_lag_{lag}'] = df['temperature_2m'].shift(lag)
    df[f'humidity_lag_{lag}'] = df['relative_humidity_2m'].shift(lag)
    df[f'wind_lag_{lag}'] = df['wind_speed_10m'].shift(lag)

# Rolling statistics
for window in [3, 6, 12, 24]:
    df[f'temp_rolling_mean_{window}'] = df['temperature_2m'].rolling(window=window).mean()
    df[f'temp_rolling_std_{window}'] = df['temperature_2m'].rolling(window=window).std()
    df[f'humidity_rolling_mean_{window}'] = df['relative_humidity_2m'].rolling(window=window).mean()

df['temperature_next_hour'] = df['temperature_2m'].shift(-1)
df = df.dropna()

features = [
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
    'precipitation', 'pressure_msl', 'cloud_cover', 'visibility',
    'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m',
    'hour', 'day', 'month', 'weekday', 'day_of_year', 'is_weekend', 'is_holiday'
]

# Add lag and rolling features
lag_features = [col for col in df.columns if 'lag_' in col or 'rolling_' in col]
features.extend(lag_features)

X = df[features]
y = df['temperature_next_hour']

# -------------------- TIME SERIES CROSS VALIDATION --------------------
tscv = TimeSeriesSplit(n_splits=5)
rmse_scores = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_scores.append(rmse)

print(f"Time Series Cross-Validation RMSE: {np.mean(rmse_scores):.4f} (±{np.std(rmse_scores):.4f})")

# -------------------- FINAL MODEL TRAINING --------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)

print("Best Random Forest Parameters:", grid_search.best_params_)
print("Random Forest → RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Random Forest → MAE:", mean_absolute_error(y_test, y_pred))
print("Random Forest → R²:", r2_score(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': best_rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Feature Importance:")
print(feature_importance.head(10))

joblib.dump(best_rf_model, "hourly_weather_model.pkl")
joblib.dump(scaler, "feature_scaler.pkl")
joblib.dump(features, "model_features.pkl")
print("Saved Random Forest model and scaler")

# -------------------- PROPHET DAILY FORECAST --------------------
df_prophet = df.rename(columns={'time': 'ds'})
features_to_forecast = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'precipitation']

all_prophet_models = {}
for feature in features_to_forecast:
    # Aggregate to daily data
    daily_df = df_prophet[['ds', feature]].copy()
    daily_df['ds'] = pd.to_datetime(daily_df['ds']).dt.date
    daily_df = daily_df.groupby('ds').mean().reset_index()
    daily_df['ds'] = pd.to_datetime(daily_df['ds'])
    daily_df = daily_df.rename(columns={feature: 'y'})
    
    # Add holidays to Prophet
    holiday_df = pd.DataFrame({
        'holiday': 'nepal_holiday',
        'ds': list(np_holidays.keys()),
        'lower_window': 0,
        'upper_window': 1,
    })
    
    # Create Prophet model with seasonality
    model = Prophet(
        daily_seasonality=True, 
        weekly_seasonality=True,
        yearly_seasonality=True,
        holidays=holiday_df
    )
    
    # Fit the model without the additional regressor that was causing the error
    model.fit(daily_df)
    all_prophet_models[feature] = model
    print(f"Trained Prophet model for {feature}")

joblib.dump(all_prophet_models, "prophet_kathmandu_all_models.pkl")
print("Saved Prophet models → prophet_kathmandu_all_models.pkl")

# -------------------- SAVE HISTORICAL DATA FOR DASHBOARD --------------------
historical_data = df[['time', 'temperature_2m', 'relative_humidity_2m', 
                      'precipitation', 'wind_speed_10m']].copy()
historical_data.to_csv("historical_weather_data.csv", index=False)
print("Saved historical data for dashboard")