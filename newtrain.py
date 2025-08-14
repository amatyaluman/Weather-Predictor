# trainhour.py - Random Forest model for weather prediction (hourly)
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class WeatherModelTrainer:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            n_jobs=-1
        )
        self.features = [
            'rain (mm)', 'precipitation (mm)', 'relative_humidity_2m',
            'wind_speed_10m (km/h)', 'wind_speed_100m (km/h)',
            'wind_direction_100m', 'weather_code',
            'hour', 'day_of_week', 'month'
        ]
        self.target = 'temperature_2m'

    def load_data(self, data_path):
        try:
            data = pd.read_csv(data_path)
            print("✅ Data loaded with columns:", data.columns.tolist())
            return data
        except Exception as e:
            raise ValueError(f"❌ Failed to load CSV: {str(e)}")

    def preprocess_data(self, data):
        print("🔍 Checking time format...")

        if 'time' not in data.columns:
            raise ValueError("'time' column is required")

        try:
            data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%dT%H:%M', errors='raise')
        except Exception:
            try:
                data['time'] = pd.to_datetime(data['time'], errors='raise')
            except Exception as e:
                raise ValueError(f"❌ Timestamp parsing failed: {e}")

        if data['time'].isnull().any():
            bad_rows = data[data['time'].isnull()]
            print("❌ Bad time values:", bad_rows.head())
            raise ValueError("Found missing/malformed time values.")

        # Extract time features
        data['hour'] = data['time'].dt.hour
        data['day_of_week'] = data['time'].dt.dayofweek
        data['month'] = data['time'].dt.month

        # Check missing features
        missing = [f for f in self.features if f not in data.columns]
        if missing:
            raise ValueError(f"❌ Missing columns: {missing}")

        # Check for null values
        if data[self.features + [self.target]].isnull().any().any():
            raise ValueError("❌ Missing values found in required columns.")

        print("✅ Data preprocessing completed.")
        return data

    def train(self, data_path, save_path=None):
        data = self.load_data(data_path)
        data = self.preprocess_data(data)

        # Train-test split
        X = data[self.features]
        y = data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("🚀 Training model...")
        self.model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"📊 RMSE on test set: {rmse:.2f}°C")

        # Show sample predictions
        print("\n📌 Sample Predictions:")
        sample_df = pd.DataFrame({'Actual': y_test[:10].values, 'Predicted': y_pred[:10]})
        print(sample_df)

        # Optional: Plot actual vs predicted
        try:
            sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
            plt.xlabel("Actual Temp (°C)")
            plt.ylabel("Predicted Temp (°C)")
            plt.title("Actual vs Predicted Temperatures")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print("⚠️ Plot skipped:", e)

        # Save model
        if save_path:
            try:
                joblib.dump(self.model, save_path)
                print(f"✅ Model saved at: {save_path}")
            except Exception as e:
                print(f"❌ Failed to save model: {str(e)}")

        return self.model

    def get_feature_importance(self):
        if not hasattr(self.model, 'feature_importances_'):
            raise Exception("Model not trained yet")
        return pd.DataFrame({
            'Feature': self.features,
            'Importance': self.model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

# === Run Training ===
if __name__ == "__main__":
    try:
        trainer = WeatherModelTrainer()
        DATA_PATH = "open-meteo-27.73N85.25E1293m.csv"
        MODEL_PATH = "weather_model.pkl"

        print("🏁 Starting model training...")
        model = trainer.train(DATA_PATH, MODEL_PATH)

        print("\n📈 Feature Importances:")
        print(trainer.get_feature_importance().to_string(index=False))

    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("📌 Please check the data file and time formatting.")
