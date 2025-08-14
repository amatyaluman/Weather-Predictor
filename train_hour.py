import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

def main():
    # Load the dataset
    df = pd.read_csv("27.6881,85.3098.csv")

    # Parse datetime
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df.dropna(subset=['datetime'], inplace=True)

    # Extract time features
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month

    # Cyclical encoding for time
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Encode location
    df['location_encoded'] = LabelEncoder().fit_transform(df['name'])

    # Drop rows with missing important values
    df.dropna(subset=['temp', 'feelslike', 'humidity', 'precip', 'windspeed', 'cloudcover', 'uvindex', 'conditions'], inplace=True)

    # Encode the weather condition (target)
    weather_encoder = LabelEncoder()
    df['weather_encoded'] = weather_encoder.fit_transform(df['conditions'])

    # Features for prediction
    features = [
        'temp', 'feelslike', 'humidity', 'precip', 'windspeed', 'cloudcover',
        'uvindex', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'location_encoded'
    ]

    X = df[features]
    y = df['weather_encoded']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_train, y_train)

    # Save the model and label encoder
    joblib.dump(model, 'hourly_weather_model.pkl')
    joblib.dump(weather_encoder, 'weather_encoder.pkl')

    # Print accuracy
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Model trained successfully with accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("27.6881,85.3098.csv")

# Drop missing values
df = df.dropna()

# Encode target column
le = LabelEncoder()
df['conditions'] = le.fit_transform(df['conditions'])

# Save label encoder for later
joblib.dump(le, "label_encoder.pkl")

# Features and target
X = df[['temp', 'feelslike', 'humidity', 'dew', 'windspeed', 'cloudcover', 'uvindex', 'visibility']]
y = df['conditions']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "hour_model.pkl")

print("Model trained and saved as hour_model.pkl")
