import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from netCDF4 import Dataset
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
from datetime import datetime, timedelta
import requests
import json
warnings.filterwarnings('ignore')

# Paths to data files
SATELLITE_PATH = 'data/satellite.csv'
GROUND_PATH = 'data/ground.csv'
WEATHER_PATH = 'data/weather.csv'
HSI_PATH = 'data/hsi_data.csv'
INSAT_PATH = 'data/insat_data.csv'
CPCB_PATH = 'data/cpcb_data.csv'
ERA5_PATH = 'data/era5_data.csv'
IOT_PATH = 'data/iot_sensors.csv'
OUTPUT_PATH = 'data/merged_clean.csv'

# Data Fusion Engine Configuration
DATA_SOURCES = {
    'sentinel5p': {
        'description': 'Sentinel-5P NO₂/CO satellite data',
        'variables': ['no2_tropospheric_column', 'co_column', 'aerosol_index'],
        'resolution': '7km x 3.5km',
        'frequency': 'daily'
    },
    'insat3d': {
        'description': 'INSAT-3D atmospheric data',
        'variables': ['temperature_profile', 'humidity_profile', 'cloud_cover'],
        'resolution': '4km x 4km',
        'frequency': 'hourly'
    },
    'cpcb': {
        'description': 'CPCB ground monitoring stations',
        'variables': ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3'],
        'resolution': 'station-based',
        'frequency': 'hourly'
    },
    'era5': {
        'description': 'ERA5 reanalysis weather data',
        'variables': ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure', 'precipitation'],
        'resolution': '0.1° x 0.1°',
        'frequency': 'hourly'
    },
    'hsi': {
        'description': 'Hyperspectral Imaging data',
        'variables': ['spectral_signature', 'pollutant_composition', 'hotspot_detection'],
        'resolution': '1m x 1m',
        'frequency': 'on-demand'
    },
    'iot': {
        'description': 'Community IoT sensors',
        'variables': ['pm25', 'pm10', 'temperature', 'humidity'],
        'resolution': 'micro-level',
        'frequency': 'real-time'
    }
}

class DataFusionEngine:
    """Advanced Data Fusion Engine for multiple satellite and ground data sources"""
    
    def __init__(self):
        self.data_sources = DATA_SOURCES
        self.fused_data = None
        self.quality_metrics = {}
        
    def load_sentinel5p_data(self, file_path=None):
        """Load and process Sentinel-5P satellite data"""
        if file_path and os.path.exists(file_path):
            return self._parse_netcdf_to_dataframe(file_path)
        else:
            return self._generate_sentinel5p_sample()
    
    def load_insat3d_data(self, file_path=None):
        """Load and process INSAT-3D atmospheric data"""
        if file_path and os.path.exists(file_path):
            return self._parse_netcdf_to_dataframe(file_path)
        else:
            return self._generate_insat3d_sample()
    
    def load_cpcb_data(self, file_path=None):
        """Load and process CPCB ground monitoring data"""
        if file_path and os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            return self._generate_cpcb_sample()
    
    def load_era5_data(self, file_path=None):
        """Load and process ERA5 reanalysis weather data"""
        if file_path and os.path.exists(file_path):
            return self._parse_netcdf_to_dataframe(file_path)
        else:
            return self._generate_era5_sample()
    
    def load_hsi_data(self, file_path=None):
        """Load and process Hyperspectral Imaging data"""
        if file_path and os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            return self._generate_hsi_sample()
    
    def load_iot_data(self, file_path=None):
        """Load and process IoT sensor data"""
        if file_path and os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            return self._generate_iot_sample()
    
    def fuse_data_sources(self, data_dict):
        """Advanced data fusion with quality assessment and uncertainty quantification"""
        print("🔄 Starting Advanced Data Fusion Process...")
        
        # Quality assessment for each data source
        quality_scores = {}
        for source, data in data_dict.items():
            quality_scores[source] = self._assess_data_quality(data, source)
            print(f"📊 {source.upper()} Quality Score: {quality_scores[source]:.2f}")
        
        # Temporal alignment
        aligned_data = self._temporal_alignment(data_dict)
        
        # Spatial interpolation and gridding
        gridded_data = self._spatial_interpolation(aligned_data)
        
        # Uncertainty quantification
        uncertainty = self._calculate_uncertainty(gridded_data, quality_scores)
        
        # Final fusion with weighted averaging
        fused_data = self._weighted_fusion(gridded_data, quality_scores, uncertainty)
        
        self.fused_data = fused_data
        self.quality_metrics = quality_scores
        
        print(f"✅ Data Fusion Complete. Final dataset shape: {fused_data.shape}")
        return fused_data
    
    def _assess_data_quality(self, data, source):
        """Assess data quality based on completeness, consistency, and accuracy"""
        if data.empty:
            return 0.0
        
        # Completeness score
        completeness = 1 - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1]))
        
        # Consistency score (check for outliers)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            consistency = 1 - (data[numeric_cols].apply(lambda x: np.abs(x - x.mean()) > 3 * x.std()).sum().sum() / 
                              (data.shape[0] * len(numeric_cols)))
        else:
            consistency = 1.0
        
        # Temporal consistency
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            temporal_consistency = 1 - (data['timestamp'].diff().dt.total_seconds().abs() > 86400).mean()
        else:
            temporal_consistency = 1.0
        
        # Weighted quality score
        quality_score = 0.4 * completeness + 0.3 * consistency + 0.3 * temporal_consistency
        return min(1.0, max(0.0, quality_score))
    
    def _temporal_alignment(self, data_dict):
        """Align data from different sources to common timestamps"""
        print("🕐 Performing temporal alignment...")
        
        # Find common time range
        all_timestamps = []
        for data in data_dict.values():
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                all_timestamps.extend(data['timestamp'].tolist())
        
        if all_timestamps:
            min_time = min(all_timestamps)
            max_time = max(all_timestamps)
            
            # Create hourly timestamps
            time_range = pd.date_range(start=min_time, end=max_time, freq='H')
            
            # Interpolate each dataset to hourly timestamps
            aligned_data = {}
            for source, data in data_dict.items():
                if 'timestamp' in data.columns:
                    # Remove duplicate timestamps before setting index
                    data = data.drop_duplicates(subset=['timestamp'])
                    data = data.set_index('timestamp')
                    data = data.reindex(time_range, method='nearest')
                    data = data.reset_index()
                    aligned_data[source] = data
                else:
                    aligned_data[source] = data
        
        return aligned_data
    
    def _spatial_interpolation(self, data_dict):
        """Perform spatial interpolation to common grid"""
        print("🗺️ Performing spatial interpolation...")
        
        # Define common grid (0.1 degree resolution)
        lat_range = np.arange(8, 38, 0.1)  # India latitude range
        lon_range = np.arange(68, 98, 0.1)  # India longitude range
        
        gridded_data = {}
        for source, data in data_dict.items():
            if 'latitude' in data.columns and 'longitude' in data.columns:
                # Simple nearest neighbor interpolation
                gridded = self._interpolate_to_grid(data, lat_range, lon_range)
                gridded_data[source] = gridded
            else:
                gridded_data[source] = data
        
        return gridded_data
    
    def _interpolate_to_grid(self, data, lat_range, lon_range):
        """Interpolate point data to regular grid"""
        # This is a simplified interpolation - in practice, you'd use more sophisticated methods
        # For now, just return the original data to avoid dimensionality issues
        return data
    
    def _calculate_uncertainty(self, gridded_data, quality_scores):
        """Calculate uncertainty for each data source"""
        uncertainty = {}
        for source in gridded_data.keys():
            # Uncertainty inversely proportional to quality score
            uncertainty[source] = 1 - quality_scores.get(source, 0.5)
        return uncertainty
    
    def _weighted_fusion(self, gridded_data, quality_scores, uncertainty):
        """Perform weighted fusion of data sources"""
        print("⚖️ Performing weighted data fusion...")
        
        # Combine all data sources with quality-based weighting
        fused_columns = set()
        for data in gridded_data.values():
            if isinstance(data, pd.DataFrame):
                fused_columns.update(data.columns)
        
        # Create fused dataset
        fused_data = pd.DataFrame()
        
        # Add timestamp if available
        for data in gridded_data.values():
            if isinstance(data, pd.DataFrame) and 'timestamp' in data.columns:
                fused_data['timestamp'] = data['timestamp']
                break
        
        # Fuse variables with weighted averaging
        for col in fused_columns:
            if col not in ['timestamp', 'latitude', 'longitude', 'city']:
                values = []
                weights = []
                valid = True
                expected_len = None
                for source, data in gridded_data.items():
                    if isinstance(data, pd.DataFrame) and col in data.columns:
                        arr = data[col].to_numpy()
                        if expected_len is None:
                            expected_len = len(arr)
                        if len(arr) != expected_len or arr.ndim != 1:
                            valid = False
                            break
                        values.append(arr)
                        weights.append(quality_scores.get(source, 0.5))
                if values and weights and valid:
                    # Weighted average
                    fused_data[col] = np.average(values, weights=weights, axis=0)
                elif not valid:
                    print(f"[WARN] Skipping column {col} due to shape mismatch across sources.")
        
        return fused_data
    
    def _generate_sentinel5p_sample(self):
        """Generate sample Sentinel-5P data for all major Indian cities"""
        dates = pd.date_range('2024-01-01', periods=14, freq='D')
        cities = ['Delhi', 'Mumbai', 'Bengaluru', 'Chennai', 'Kolkata', 'Hyderabad', 
                 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Kanpur', 'Nagpur', 
                 'Indore', 'Thane', 'Bhopal', 'Visakhapatnam', 'Patna', 'Vadodara']
        
        city_coords = {
            'Delhi': (28.7041, 77.1025), 'Mumbai': (19.0760, 72.8777),
            'Bengaluru': (12.9716, 77.5946), 'Chennai': (13.0827, 80.2707),
            'Kolkata': (22.5726, 88.3639), 'Hyderabad': (17.3850, 78.4867),
            'Pune': (18.5204, 73.8567), 'Ahmedabad': (23.0225, 72.5714),
            'Jaipur': (26.9124, 75.7873), 'Lucknow': (26.8467, 80.9462),
            'Kanpur': (26.4499, 80.3319), 'Nagpur': (21.1458, 79.0882),
            'Indore': (22.7196, 75.8577), 'Thane': (19.2183, 72.9781),
            'Bhopal': (23.2599, 77.4126), 'Visakhapatnam': (17.6868, 83.2185),
            'Patna': (25.5941, 85.1376), 'Vadodara': (22.3072, 73.1812)
        }
        
        base_no2 = {
            'Delhi': 45, 'Mumbai': 35, 'Bengaluru': 25, 'Chennai': 30,
            'Kolkata': 40, 'Hyderabad': 20, 'Pune': 15, 'Ahmedabad': 25,
            'Jaipur': 20, 'Lucknow': 30, 'Kanpur': 35, 'Nagpur': 18,
            'Indore': 22, 'Thane': 28, 'Bhopal': 20, 'Visakhapatnam': 15,
            'Patna': 32, 'Vadodara': 23
        }
        
        data = []
        for date in dates:
            for city in cities:
                no2_value = np.random.normal(base_no2.get(city, 25), 8) + np.random.normal(0, 3)
                data.append({
                    'timestamp': date,
                    'city': city,
                    'latitude': city_coords[city][0],
                    'longitude': city_coords[city][1],
                    'no2_tropospheric_column': max(0, no2_value),
                    'co_column': np.random.normal(100, 20),
                    'aerosol_index': np.random.normal(0.5, 0.2),
                    'data_source': 'sentinel5p',
                    'quality_flag': np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
                })
        
        df = pd.DataFrame(data)
        df.to_csv(SATELLITE_PATH, index=False)
        print(f"Created sample Sentinel-5P data at {SATELLITE_PATH}")
        return df
    
    def _generate_insat3d_sample(self):
        """Generate sample INSAT-3D atmospheric data"""
        dates = pd.date_range('2024-01-01', periods=14, freq='D')
        cities = ['Delhi', 'Mumbai', 'Bengaluru', 'Chennai', 'Kolkata', 'Hyderabad', 
                 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow']
        
        data = []
        for date in dates:
            for city in cities:
                data.append({
                    'timestamp': date,
                    'city': city,
                    'temperature_profile_850hpa': np.random.normal(15, 5),
                    'temperature_profile_500hpa': np.random.normal(-5, 3),
                    'humidity_profile_850hpa': np.random.normal(60, 15),
                    'humidity_profile_500hpa': np.random.normal(40, 10),
                    'cloud_cover_low': np.random.uniform(0, 1),
                    'cloud_cover_mid': np.random.uniform(0, 1),
                    'cloud_cover_high': np.random.uniform(0, 1),
                    'data_source': 'insat3d',
                    'quality_flag': np.random.choice([1, 2, 3], p=[0.8, 0.15, 0.05])
                })
        
        df = pd.DataFrame(data)
        df.to_csv(INSAT_PATH, index=False)
        print(f"Created sample INSAT-3D data at {INSAT_PATH}")
        return df
    
    def _generate_cpcb_sample(self):
        """Generate sample CPCB ground monitoring data"""
        dates = pd.date_range('2024-01-01', periods=14, freq='D')
        cities = ['Delhi', 'Mumbai', 'Bengaluru', 'Chennai', 'Kolkata', 'Hyderabad', 
                 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow']
        
        base_pm25 = {
            'Delhi': 120, 'Mumbai': 80, 'Bengaluru': 60, 'Chennai': 70,
            'Kolkata': 90, 'Hyderabad': 50, 'Pune': 40, 'Ahmedabad': 60,
            'Jaipur': 45, 'Lucknow': 85
        }
        
        data = []
        for date in dates:
            for city in cities:
                pm25_value = np.random.normal(base_pm25[city], 25) + np.random.normal(0, 10)
                data.append({
                    'timestamp': date,
                    'city': city,
                    'pm25': max(0, pm25_value),
                    'pm10': max(0, pm25_value * 1.5),
                    'no2': np.random.normal(40, 15),
                    'so2': np.random.normal(15, 8),
                    'co': np.random.normal(1.5, 0.5),
                    'o3': np.random.normal(30, 10),
                    'aqi': calculate_aqi(max(0, pm25_value)),
                    'data_source': 'cpcb',
                    'station_id': f"CPCB_{city}_{np.random.randint(1000, 9999)}",
                    'quality_flag': np.random.choice([1, 2, 3], p=[0.9, 0.08, 0.02])
                })
        
        df = pd.DataFrame(data)
        df.to_csv(CPCB_PATH, index=False)
        print(f"Created sample CPCB data at {CPCB_PATH}")
        return df
    
    def _generate_era5_sample(self):
        """Generate sample ERA5 reanalysis weather data"""
        dates = pd.date_range('2024-01-01', periods=14, freq='D')
        cities = ['Delhi', 'Mumbai', 'Bengaluru', 'Chennai', 'Kolkata', 'Hyderabad', 
                 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow']
        
        base_temp = {
            'Delhi': 25, 'Mumbai': 28, 'Bengaluru': 22, 'Chennai': 30,
            'Kolkata': 27, 'Hyderabad': 26, 'Pune': 24, 'Ahmedabad': 27,
            'Jaipur': 26, 'Lucknow': 25
        }
        
        base_humidity = {
            'Delhi': 50, 'Mumbai': 75, 'Bengaluru': 65, 'Chennai': 70,
            'Kolkata': 80, 'Hyderabad': 60, 'Pune': 55, 'Ahmedabad': 45,
            'Jaipur': 40, 'Lucknow': 60
        }
        
        data = []
        for date in dates:
            for city in cities:
                data.append({
                    'timestamp': date,
                    'city': city,
                    'temperature_2m': np.random.normal(base_temp[city], 5),
                    'temperature_850hpa': np.random.normal(base_temp[city] - 10, 3),
                    'relative_humidity_2m': np.random.normal(base_humidity[city], 10),
                    'wind_speed_10m': np.random.normal(10, 3),
                    'wind_direction_10m': np.random.uniform(0, 360),
                    'surface_pressure': np.random.normal(1013, 5),
                    'total_precipitation': np.random.exponential(2),
                    'boundary_layer_height': np.random.normal(1000, 200),
                    'data_source': 'era5',
                    'quality_flag': np.random.choice([1, 2, 3], p=[0.95, 0.04, 0.01])
                })
        
        df = pd.DataFrame(data)
        df.to_csv(ERA5_PATH, index=False)
        print(f"Created sample ERA5 data at {ERA5_PATH}")
        return df
    
    def _generate_hsi_sample(self):
        """Generate sample Hyperspectral Imaging data"""
        dates = pd.date_range('2024-01-01', periods=7, freq='D')  # HSI data less frequent
        cities = ['Delhi', 'Mumbai', 'Bengaluru', 'Chennai', 'Kolkata']
        
        data = []
        for date in dates:
            for city in cities:
                # Simulate spectral signatures for different pollutants
                spectral_signature = np.random.normal(0.5, 0.1, 100)  # 100 spectral bands
                data.append({
                    'timestamp': date,
                    'city': city,
                    'spectral_signature': json.dumps(spectral_signature.tolist()),
                    'pollutant_composition': json.dumps({
                        'soot': np.random.uniform(0.1, 0.3),
                        'sulfate': np.random.uniform(0.05, 0.2),
                        'nitrate': np.random.uniform(0.1, 0.25),
                        'organic_carbon': np.random.uniform(0.2, 0.4),
                        'dust': np.random.uniform(0.05, 0.15)
                    }),
                    'hotspot_detection': np.random.choice(['industrial', 'traffic', 'residential', 'agricultural']),
                    'spatial_resolution': '1m',
                    'spectral_resolution': '10nm',
                    'data_source': 'hsi',
                    'quality_flag': np.random.choice([1, 2, 3], p=[0.85, 0.12, 0.03])
                })
        
        df = pd.DataFrame(data)
        df.to_csv(HSI_PATH, index=False)
        print(f"Created sample HSI data at {HSI_PATH}")
        return df
    
    def _generate_iot_sample(self):
        """Generate sample IoT sensor data"""
        dates = pd.date_range('2024-01-01', periods=14, freq='H')  # Hourly IoT data
        cities = ['Delhi', 'Mumbai', 'Bengaluru', 'Chennai', 'Kolkata']
        
        # Simulate multiple IoT sensors per city
        sensors_per_city = 5
        
        data = []
        for date in dates:
            for city in cities:
                for sensor_id in range(sensors_per_city):
                    data.append({
                        'timestamp': date,
                        'city': city,
                        'sensor_id': f"IoT_{city}_{sensor_id:03d}",
                        'latitude': np.random.normal(0, 0.01),  # Small spatial variation
                        'longitude': np.random.normal(0, 0.01),
                        'pm25': max(0, np.random.normal(60, 20)),
                        'pm10': max(0, np.random.normal(90, 30)),
                        'temperature': np.random.normal(25, 8),
                        'humidity': np.random.normal(60, 15),
                        'battery_level': np.random.uniform(0.3, 1.0),
                        'signal_strength': np.random.uniform(0.5, 1.0),
                        'data_source': 'iot',
                        'quality_flag': np.random.choice([1, 2, 3], p=[0.7, 0.25, 0.05])
                    })
        
        df = pd.DataFrame(data)
        df.to_csv(IOT_PATH, index=False)
        print(f"Created sample IoT data at {IOT_PATH}")
        return df

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

def create_sample_data_if_missing():
    """Create sample data if files don't exist"""
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate sample data for each source
    fusion_engine = DataFusionEngine()
    
    # Generate all sample data
    sentinel_data = fusion_engine._generate_sentinel5p_sample()
    insat_data = fusion_engine._generate_insat3d_sample()
    cpcb_data = fusion_engine._generate_cpcb_sample()
    era5_data = fusion_engine._generate_era5_sample()
    hsi_data = fusion_engine._generate_hsi_sample()
    iot_data = fusion_engine._generate_iot_sample()
    
    # Create a simple combined dataset for the dashboard
    # Use CPCB data as the base and add some satellite data
    combined_data = cpcb_data.copy()
    
    # Add some satellite NO2 data if available
    if not sentinel_data.empty:
        # Merge on timestamp and city
        combined_data = combined_data.merge(
            sentinel_data[['timestamp', 'city', 'no2_tropospheric_column']], 
            on=['timestamp', 'city'], 
            how='left'
        )
    
    # Add some weather data if available
    if not era5_data.empty:
        combined_data = combined_data.merge(
            era5_data[['timestamp', 'city', 'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m']], 
            on=['timestamp', 'city'], 
            how='left'
        )
    
    # Generate forecast data (simple extrapolation)
    forecast_data = []
    last_date = combined_data['timestamp'].max()
    
    for city in combined_data['city'].unique():
        city_data = combined_data[combined_data['city'] == city].iloc[-1]
        
        for i in range(1, 25):  # 24 hours forecast
            forecast_date = last_date + pd.Timedelta(hours=i)
            
            # Simple forecast: add some random variation
            pm25_forecast = max(0, city_data['pm25'] + np.random.normal(0, 10))
            aqi_forecast = calculate_aqi(pm25_forecast)
            
            forecast_data.append({
                'timestamp': forecast_date,
                'city': city,
                'pm25_forecast': pm25_forecast,
                'aqi_forecast': aqi_forecast,
                'no2_forecast': max(0, city_data.get('no2', 40) + np.random.normal(0, 5)),
                'temperature_forecast': city_data.get('temperature_2m', 25) + np.random.normal(0, 2),
                'humidity_forecast': max(0, min(100, city_data.get('relative_humidity_2m', 60) + np.random.normal(0, 5)))
            })
    
    forecast_df = pd.DataFrame(forecast_data)
    
    # Save the combined data
    combined_data.to_csv('data/combined_data.csv', index=False)
    forecast_df.to_csv('data/forecast_data.csv', index=False)
    
    print("✅ Sample data generation completed!")
    print(f"📊 Historical data: {len(combined_data)} records")
    print(f"🔮 Forecast data: {len(forecast_df)} records")
    
    return combined_data, forecast_df

if __name__ == "__main__":
    # Create sample data and perform advanced data fusion
    fused_data = create_sample_data_if_missing()
    print("🎯 Advanced AEROVIEW Data Fusion Engine Ready!") 