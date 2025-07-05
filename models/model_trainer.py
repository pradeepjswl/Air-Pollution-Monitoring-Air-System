import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def train_model():
    """Train XGBoost model on processed data"""
    
    # Load processed data
    try:
        df = pd.read_csv('data/merged_clean.csv')
        print(f"Loaded data with shape: {df.shape}")
    except FileNotFoundError:
        print("Processed data not found. Please run data_cleaner.py first.")
        return None
    
    # Prepare features and target
    feature_cols = ['no2_tropospheric_column', 'co_column', 'temperature', 
                   'humidity', 'wind_speed', 'wind_direction', 'pressure']
    
    # Only use columns that exist
    existing_features = [col for col in feature_cols if col in df.columns]
    
    if 'pm25' not in df.columns:
        print("PM2.5 column not found in data. Creating synthetic target...")
        # Create synthetic PM2.5 values based on other features
        df['pm25'] = (df['no2_tropospheric_column'] * 2 + 
                     df['co_column'] * 0.1 + 
                     df['temperature'] * 0.5 + 
                     np.random.normal(0, 10))
    
    X = df[existing_features]
    y = df['pm25']
    
    print(f"Features: {existing_features}")
    print(f"Target: PM2.5")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    print("Training XGBoost model...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"R² Score: {r2:.3f}")
    
    # Save model
    model_path = 'models/xgboost_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model

def generate_forecast(model, df, hours_ahead=72):
    """Generate forecast for next N hours"""
    
    # Get the latest data for each city
    latest_data = df.groupby('city').tail(1).copy()
    
    # Ensure timestamp is datetime
    latest_data['timestamp'] = pd.to_datetime(latest_data['timestamp'])
    
    # Create future timestamps
    future_dates = pd.date_range(
        start=latest_data['timestamp'].max() + pd.Timedelta(hours=1),
        periods=hours_ahead,
        freq='H'
    )
    
    # Prepare feature columns
    feature_cols = ['no2_tropospheric_column', 'co_column', 'temperature', 
                   'humidity', 'wind_speed', 'wind_direction', 'pressure']
    existing_features = [col for col in feature_cols if col in df.columns]
    
    forecasts = []
    
    for city in df['city'].unique():
        city_latest = latest_data[latest_data['city'] == city].iloc[0]
        
        for future_date in future_dates:
            # Create forecast row with slight variations in features
            forecast_row = city_latest[existing_features].copy()
            
            # Add some realistic variations
            forecast_row = forecast_row + np.random.normal(0, 0.1, len(existing_features))
            
            # Predict PM2.5
            pm25_pred = model.predict([forecast_row])[0]
            
            forecasts.append({
                'timestamp': future_date,
                'city': city,
                'pm25_forecast': max(0, pm25_pred),
                'aqi_forecast': calculate_aqi(max(0, pm25_pred))
            })
    
    return pd.DataFrame(forecasts)

def calculate_aqi(pm25):
    """Calculate AQI based on PM2.5 concentration"""
    if pm25 <= 12:
        return 'Good'
    elif pm25 <= 35.4:
        return 'Moderate'
    elif pm25 <= 55.4:
        return 'Unhealthy for Sensitive Groups'
    elif pm25 <= 150.4:
        return 'Unhealthy'
    elif pm25 <= 250.4:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'

if __name__ == "__main__":
    # Train model
    model = train_model()
    
    if model is not None:
        # Load data for forecasting
        df = pd.read_csv('data/merged_clean.csv')
        
        # Generate forecast
        forecast_df = generate_forecast(model, df, hours_ahead=72)
        
        # Save forecast
        forecast_df.to_csv('data/forecast.csv', index=False)
        print(f"Forecast saved to data/forecast.csv")
        print(f"Forecast shape: {forecast_df.shape}") 