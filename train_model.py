import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import accuracy_score

# Create weather labels from existing data
def create_weather_label(row):
    if row['Rainfall'] > 5.0:
        return 'Rainy'
    elif row['MaxTemp'] > 30:
        return 'Hot'
    elif row['Humidity'] > 80:
        return 'Humid'
    elif row['WS10M_MAX'] > 30:
        return 'Windy'
    else:
        return 'Normal'

# Load and prepare data
df = pd.read_csv('weather_data.csv')

# Create target variable
df['Weather'] = df.apply(create_weather_label, axis=1)

# Feature engineering
df['Date'] = pd.to_datetime(df['Date'])
df['DayOfYear'] = df['Date'].dt.dayofyear
df['MonthSin'] = np.sin(2 * np.pi * df['Month']/12)
df['MonthCos'] = np.cos(2 * np.pi * df['Month']/12)

# Encode categorical features
district_encoder = LabelEncoder()
weather_encoder = LabelEncoder()
df['DistrictEncoded'] = district_encoder.fit_transform(df['District'])
df['WeatherEncoded'] = weather_encoder.fit_transform(df['Weather'])

# Select features from original columns
features = [
    'MonthSin', 'MonthCos', 'DayOfYear', 'DistrictEncoded',
    'Rainfall', 'Pressure', 'Humidity', 'AvgTemp',
    'MaxTemp', 'MinTemp', 'TempRange', 'WindSpeed',
    'WS10M_MAX', 'WS10M_RANGE', 'WS50M_MAX'
]

X = df[features]
y = df['WeatherEncoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save artifacts
joblib.dump(model, 'weather_model.pkl')
joblib.dump(district_encoder, 'district_encoder.pkl')
joblib.dump(weather_encoder, 'weather_encoder.pkl')

print("Model trained and saved successfully!")