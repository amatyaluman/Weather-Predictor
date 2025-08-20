#train_weather_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# -------------------- CSV FILE --------------------
csv_file = "KTMLTP.csv"

# -------------------- READ COMMA-SEPARATED CSV --------------------
print("Reading comma-separated CSV...")

# Read the entire file to understand its structure
with open(csv_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Split into lines
lines = content.split('\n')

# Find where the data sections start
section_starts = []
for i, line in enumerate(lines):
    if 'location_id' in line and ('time' in line or 'latitude' in line):
        section_starts.append(i)

print(f"Found section starts at lines: {section_starts}")

# Read each section separately
dfs = []
section_names = ['location', 'hourly', 'enhanced_hourly', 'daily']

for i, start_line in enumerate(section_starts):
    end_line = section_starts[i+1] if i < len(section_starts)-1 else None
    
    print(f"Reading {section_names[i]} section from line {start_line} to {end_line}")
    
    # Extract the lines for this section
    section_lines = lines[start_line:end_line] if end_line else lines[start_line:]
    
    # Parse the header
    header = section_lines[0].split(',')
    print(f"  Header has {len(header)} columns: {header[:3]}...")  # Show first 3 columns
    
    # Parse data rows
    data_rows = []
    for line in section_lines[1:]:
        if line.strip():  # Skip empty lines
            values = line.split(',')
            if len(values) == len(header):
                data_rows.append(values)
    
    # Create DataFrame
    if data_rows:
        df_section = pd.DataFrame(data_rows, columns=header)
        print(f"  {section_names[i]} section: {df_section.shape}")
        dfs.append(df_section)
    else:
        print(f"  No data rows found for {section_names[i]} section")

# -------------------- DATA PREPROCESSING --------------------
def clean_and_preprocess(df, section_type):
    """Clean and preprocess the dataframe"""
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    # Clean column names
    df_clean.columns = [str(col).strip().lower().replace(' ', '_').replace('(', '').replace(')', '')
                       .replace('°', '').replace('Â', '').replace('Ã', '').replace('`', '')
                       .replace(')', '').replace('(', '').replace('/', '_per_')
                       .replace('³', '3').replace('²', '2')  # Handle superscripts
                       for col in df_clean.columns]
    
    # Convert numeric columns
    for col in df_clean.columns:
        # Skip obvious non-numeric columns
        if any(x in col for x in ['time', 'date', 'id', 'code', 'name', 'zone', 'sunrise', 'sunset']):
            continue
            
        try:
            # Try converting to numeric
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        except:
            # If conversion fails, check if it might be datetime
            try:
                if df_clean[col].dtype == 'object' and any(char in str(df_clean[col].iloc[0]) for char in ['-', ':', 'T']):
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            except:
                pass
    
    # Find datetime columns and extract features
    datetime_cols = []
    for col in df_clean.columns:
        if pd.api.types.is_datetime64_any_dtype(df_clean[col]):
            datetime_cols.append(col)
            # Extract time features
            df_clean[f'{col}_hour'] = df_clean[col].dt.hour
            df_clean[f'{col}_day'] = df_clean[col].dt.day
            df_clean[f'{col}_month'] = df_clean[col].dt.month
            df_clean[f'{col}_year'] = df_clean[col].dt.year
            df_clean[f'{col}_dayofweek'] = df_clean[col].dt.dayofweek
    
    print(f"  Processed {section_type}: {df_clean.shape}, Numeric columns: {len(df_clean.select_dtypes(include=[np.number]).columns)}")
    return df_clean

# Process the dataframes
processed_dfs = []
for i, df in enumerate(dfs):
    if not df.empty:
        processed_df = clean_and_preprocess(df, section_names[i])
        processed_dfs.append(processed_df)

# Use the main data sections for training
hourly_df = pd.DataFrame()
daily_df = pd.DataFrame()

for i, df in enumerate(processed_dfs):
    if section_names[i] == 'hourly' and not df.empty:
        hourly_df = df
    elif section_names[i] == 'daily' and not df.empty:
        daily_df = df

print(f"\nHourly data: {hourly_df.shape}")
print(f"Daily data: {daily_df.shape}")

# -------------------- IMPROVED MODEL TRAINING --------------------
def train_improved_model(df, model_type='hourly'):
    """Improved model training with better data handling"""
    if df.empty:
        print(f"No {model_type} data for training")
        return None
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        print(f"No numeric columns in {model_type} data")
        return None
    
    print(f"{model_type.capitalize()} data has {len(numeric_cols)} numeric columns")
    
    # Identify target columns
    if model_type == 'hourly':
        target_keywords = ['temperature', 'humidity', 'precipitation', 'wind', 'pressure']
    else:
        target_keywords = ['max', 'min', 'mean', 'sum', 'total']
    
    target_cols = []
    for col in numeric_cols:
        if any(keyword in col for keyword in target_keywords):
            non_null = df[col].notna().sum()
            if non_null > 50:
                target_cols.append(col)
                print(f"  Target: {col} ({non_null} values)")
    
    if not target_cols:
        print(f"No suitable targets found, using first 3 numeric columns")
        target_cols = numeric_cols[:3]
    
    # For each target, select the most relevant features
    models = {}
    
    for target in target_cols:
        print(f"\nTraining model for: {target}")
        
        # Select features that are correlated with the target
        # First, get all other numeric columns
        potential_features = [col for col in numeric_cols if col != target]
        
        # Calculate correlation with target (handle missing values)
        correlations = {}
        for feature in potential_features:
            valid_idx = df[[target, feature]].notna().all(axis=1)
            if valid_idx.sum() > 50:  # Need enough data to calculate correlation
                corr = df.loc[valid_idx, target].corr(df.loc[valid_idx, feature])
                if not pd.isna(corr):
                    correlations[feature] = abs(corr)
        
        # Select top 10 most correlated features, or all if fewer than 10
        if correlations:
            feature_cols = sorted(correlations, key=correlations.get, reverse=True)[:10]
            print(f"  Selected {len(feature_cols)} features based on correlation")
        else:
            # Fallback: use all available features
            feature_cols = potential_features
            print(f"  Using all {len(feature_cols)} available features")
        
        if not feature_cols:
            print(f"  No feature columns available for {target}")
            continue
        
        # Prepare data - handle missing values more gracefully
        valid_mask = df[target].notna()
        for feature in feature_cols:
            valid_mask = valid_mask & df[feature].notna()
        
        if valid_mask.sum() < 20:
            print(f"  Not enough complete data: {valid_mask.sum()} samples")
            continue
        
        X = df.loc[valid_mask, feature_cols]
        y = df.loc[valid_mask, target]
        
        print(f"  Training with {len(X)} samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        try:
            model = RandomForestRegressor(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1,
                max_depth=10
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            
            # Store model in dictionary with a unique key
            model_key = f"{model_type}_{target}"
            models[model_key] = model
            print(f"  Model stored for {model_key}")
            
        except Exception as e:
            print(f"  Error training {target}: {e}")
    
    return models

# -------------------- SIMPLE MODEL TRAINING (FALLBACK) --------------------
def train_simple_model(df, model_type='hourly'):
    """Simple model training as fallback"""
    if df.empty:
        return None
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return None
    
    # Use the first column as target, second as feature
    target = numeric_cols[0]
    feature = numeric_cols[1]
    
    print(f"Training simple model: {feature} -> {target}")
    
    # Prepare data
    valid_mask = df[[target, feature]].notna().all(axis=1)
    if valid_mask.sum() < 10:
        print("Not enough data for simple model")
        return None
    
    X = df.loc[valid_mask, [feature]]
    y = df.loc[valid_mask, target]
    
    # Train model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Evaluate
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print(f"Simple model RMSE: {rmse:.4f}")
    
    # Store model in dictionary with a unique key
    model_key = f"simple_{model_type}_{target}"
    print(f"Simple model stored for {model_key}")
    
    return {model_key: model}

# -------------------- TRAIN MODELS --------------------
print("\n" + "="*60)
print("TRAINING IMPROVED HOURLY MODELS")
print("="*60)
hourly_models = train_improved_model(hourly_df, 'hourly')

print("\n" + "="*60)
print("TRAINING IMPROVED DAILY MODELS")
print("="*60)
daily_models = train_improved_model(daily_df, 'daily')

# If improved models failed, try simple models
if not hourly_models:
    print("\nTrying simple hourly model...")
    hourly_models = train_simple_model(hourly_df, 'hourly')

if not daily_models:
    print("\nTrying simple daily model...")
    daily_models = train_simple_model(daily_df, 'daily')

# -------------------- COMBINE AND SAVE MODELS --------------------
print("\n" + "="*60)
print("COMBINING AND SAVING ALL MODELS")
print("="*60)

# Combine all models into a single dictionary
all_models = {}
if hourly_models:
    all_models.update(hourly_models)
if daily_models:
    all_models.update(daily_models)

# Save all models to a single .pkl file
if all_models:
    model_file = "all_weather_models.pkl"
    joblib.dump(all_models, model_file)
    print(f"All models saved to {model_file}")
else:
    print("No models to save")

# -------------------- RESULTS --------------------
print("\n" + "="*60)
print("FINAL TRAINING RESULTS")
print("="*60)

if hourly_models:
    print(f"✓ Trained {len(hourly_models)} hourly models")
    for target in hourly_models.keys():
        print(f"  - {target}")
else:
    print("✗ No hourly models trained")

if daily_models:
    print(f"✓ Trained {len(daily_models)} daily models")
    for target in daily_models.keys():
        print(f"  - {target}")
else:
    print("✗ No daily models trained")

# Debug information
print(f"\nDebug information:")
print(f"Hourly data shape: {hourly_df.shape}")
print(f"Hourly numeric columns: {len(hourly_df.select_dtypes(include=[np.number]).columns)}")
if not hourly_df.empty:
    print(f"Hourly data sample:")
    print(hourly_df.select_dtypes(include=[np.number]).iloc[:3, :5])

print(f"\nDaily data shape: {daily_df.shape}")
print(f"Daily numeric columns: {len(daily_df.select_dtypes(include=[np.number]).columns)}")
if not daily_df.empty:
    print(f"Daily data sample:")
    print(daily_df.select_dtypes(include=[np.number]).iloc[:3, :5])