# train_weather_model.py
# Multi-target Random Forest model for weather prediction (temperature + wind speed)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import joblib

class WeatherModelTrainer:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        self.features = [
            'rain (mm)', 'precipitation (mm)', 'relative_humidity_2m',
            'wind_speed_100m (km/h)', 'wind_direction_100m', 'weather_code',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
        ]
        # Multiple targets now
        self.targets = ['temperature_2m', 'wind_speed_10m (km/h)']

    def load_data(self, data_path):
        data = pd.read_csv(data_path)
        print("Data loaded with shape:", data.shape)
        return data

    def preprocess_data(self, data):
        data['time'] = pd.to_datetime(data['time'])
        data['hour'] = data['time'].dt.hour
        data['month'] = data['time'].dt.month

        data['hour_sin'] = np.sin(2 * np.pi * data['hour']/24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour']/24)
        data['month_sin'] = np.sin(2 * np.pi * data['month']/12)
        data['month_cos'] = np.cos(2 * np.pi * data['month']/12)

        bins = [0, 1, 3, 50, 70, 100]
        labels = ['clear', 'cloudy', 'fog', 'rain', 'storm']
        data['weather_type'] = pd.cut(data['weather_code'], bins=bins, labels=labels)
        data = pd.get_dummies(data, columns=['weather_type'])

        return data

    def train(self, data_path, save_path=None):
        data = self.load_data(data_path)
        data = self.preprocess_data(data)

        X = data[self.features]
        y = data[self.targets]   # multiple targets
        tscv = TimeSeriesSplit(n_splits=5)

        rmse_scores = {target: [] for target in self.targets}

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            # y_pred is 2D: [n_samples, n_targets]
            for i, target in enumerate(self.targets):
                rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
                rmse_scores[target].append(rmse)

        print("Average RMSE across folds:")
        for target in self.targets:
            print(f"  {target}: {np.mean(rmse_scores[target]):.2f}")

        # Full training
        self.model.fit(X, y)

        if save_path:
            joblib.dump(self.model, save_path)
            print(f"Model saved to {save_path}")

        return self.model

if __name__ == "__main__":
    trainer = WeatherModelTrainer()
    trainer.train(
        data_path="open-meteo-27.73N85.25E1293m.csv",
        save_path="optimized_weather_model.pkl"
    )
