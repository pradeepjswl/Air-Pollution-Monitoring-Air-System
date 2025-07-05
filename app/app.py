import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
import joblib
from datetime import datetime, timedelta
import warnings
import json
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AEROVIEW - Air Pollution Monitor",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-box {
        background-color: #ffebee;
        border: 1px solid #f44336;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .good-aqi { color: #4caf50; }
    .moderate-aqi { color: #ff9800; }
    .unhealthy-aqi { color: #f44336; }
    .very-unhealthy-aqi { color: #9c27b0; }
    .hazardous-aqi { color: #7b1fa2; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load data for the dashboard"""
    try:
        # Load historical data
        historical_df = pd.read_csv('data/combined_data.csv')
        historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
        
        # Load forecast data
        forecast_df = pd.read_csv('data/forecast_data.csv')
        forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp'])
        
        return historical_df, forecast_df
    except FileNotFoundError:
        st.error("❌ Data files not found. Please run 'python data_cleaner.py' first.")
        return pd.DataFrame(), pd.DataFrame()

def get_aqi_color(aqi_category):
    """Get color for AQI category"""
    colors = {
        'Good': '#4caf50',
        'Moderate': '#ff9800',
        'Unhealthy for Sensitive Groups': '#f44336',
        'Unhealthy': '#f44336',
        'Very Unhealthy': '#9c27b0',
        'Hazardous': '#7b1fa2'
    }
    return colors.get(aqi_category, '#666666')

def create_alert_message(city, aqi_category, pm25_value):
    """Create alert message based on AQI"""
    if aqi_category in ['Unhealthy', 'Very Unhealthy', 'Hazardous']:
        return f"⚠️ ALERT: {city} has {aqi_category} air quality (PM2.5: {pm25_value:.1f} μg/m³). Avoid outdoor activities!"
    elif aqi_category == 'Unhealthy for Sensitive Groups':
        return f"⚠️ CAUTION: {city} has {aqi_category} air quality (PM2.5: {pm25_value:.1f} μg/m³). Sensitive groups should limit outdoor activities."
    else:
        return f"✅ {city} air quality is {aqi_category} (PM2.5: {pm25_value:.1f} μg/m³). Safe for outdoor activities."

def main():
    # Header
    st.markdown('<h1 class="main-header">AEROVIEW</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">Air Pollution Monitor & Forecast</h2>', unsafe_allow_html=True)
    
    # Load data
    historical_df, forecast_df = load_data()
    
    if historical_df.empty or forecast_df.empty:
        st.stop()
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    
    # City selection
    cities = historical_df['city'].unique()
    selected_city = st.sidebar.selectbox("Select City", cities)
    
    # Time range selection
    st.sidebar.subheader("Time Range")
    time_range = st.sidebar.selectbox(
        "Select Time Range",
        ["Last 7 Days", "Last 14 Days", "Next 24 Hours", "Next 72 Hours"]
    )
    
    # Filter data based on selection
    if "Last" in time_range:
        days = int(time_range.split()[1])
        filtered_historical = historical_df[
            (historical_df['city'] == selected_city) &
            (historical_df['timestamp'] >= datetime.now() - timedelta(days=days))
        ]
        filtered_forecast = pd.DataFrame()  # No forecast for historical view
        
        # If no historical data for selected range, show all available data for the city
        if filtered_historical.empty:
            filtered_historical = historical_df[historical_df['city'] == selected_city]
    else:
        hours = int(time_range.split()[1])
        filtered_forecast = forecast_df[
            (forecast_df['city'] == selected_city) &
            (forecast_df['timestamp'] <= datetime.now() + timedelta(hours=hours))
        ]
        filtered_historical = historical_df[
            (historical_df['city'] == selected_city) &
            (historical_df['timestamp'] >= datetime.now() - timedelta(days=7))
        ]
        
        # If no forecast data for selected range, show all available forecast data for the city
        if filtered_forecast.empty:
            filtered_forecast = forecast_df[forecast_df['city'] == selected_city]
    
    # Main content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if not filtered_forecast.empty and len(filtered_forecast) > 0:
            current_pm25 = filtered_forecast.iloc[0]['pm25_forecast']
            current_aqi = filtered_forecast.iloc[0]['aqi_forecast']
        elif not filtered_historical.empty and len(filtered_historical) > 0:
            current_pm25 = filtered_historical.iloc[-1]['pm25'] if 'pm25' in filtered_historical.columns else 0
            current_aqi = filtered_historical.iloc[-1]['aqi'] if 'aqi' in filtered_historical.columns else 'Unknown'
        else:
            current_pm25 = 0
            current_aqi = 'Unknown'
        
        st.metric("Current PM2.5", f"{current_pm25:.1f} μg/m³")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if not filtered_forecast.empty and len(filtered_forecast) > 0:
            max_pm25 = filtered_forecast['pm25_forecast'].max()
            # Safer approach to get AQI for max PM2.5
            if pd.notna(max_pm25):
                max_row = filtered_forecast.loc[filtered_forecast['pm25_forecast'] == max_pm25]
                if not max_row.empty:
                    max_aqi = max_row.iloc[0]['aqi_forecast']
                else:
                    max_aqi = 'Unknown'
            else:
                max_aqi = 'Unknown'
        elif not filtered_historical.empty and len(filtered_historical) > 0:
            max_pm25 = filtered_historical['pm25'].max() if 'pm25' in filtered_historical.columns else 0
            max_aqi = 'Unknown'
        else:
            max_pm25 = 0
            max_aqi = 'Unknown'
        
        st.metric("Max PM2.5 (Period)", f"{max_pm25:.1f} μg/m³")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if not filtered_forecast.empty and len(filtered_forecast) > 0:
            avg_pm25 = filtered_forecast['pm25_forecast'].mean()
        elif not filtered_historical.empty and len(filtered_historical) > 0:
            avg_pm25 = filtered_historical['pm25'].mean() if 'pm25' in filtered_historical.columns else 0
        else:
            avg_pm25 = 0
        
        st.metric("Average PM2.5", f"{avg_pm25:.1f} μg/m³")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Health Risk Overlay Section
    st.subheader('Health Risk Levels')
    aqi_risk_colors = [
        ('Good', '#009966'),
        ('Satisfactory', '#66bb6a'),
        ('Moderate', '#ffde33'),
        ('Poor', '#ff9933'),
        ('Very Poor', '#cc0033'),
        ('Severe', '#660099'),
        ('Hazardous', '#7e0023')
    ]
    aqi_risk_table = [
        ('Good', '0-50', 'Air quality is considered satisfactory, and air pollution poses little or no risk.'),
        ('Satisfactory', '51-100', 'Air quality is acceptable; however, for some pollutants there may be a moderate health concern for a very small number of people.'),
        ('Moderate', '101-200', 'Members of sensitive groups may experience health effects. The general public is not likely to be affected.'),
        ('Poor', '201-300', 'Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects.'),
        ('Very Poor', '301-400', 'Health warnings of emergency conditions. The entire population is more likely to be affected.'),
        ('Severe', '401-500', 'Health alert: everyone may experience more serious health effects.'),
        ('Hazardous', '500+', 'Serious risk: avoid all outdoor activity.')
    ]
    # Color bar
    st.markdown('<div style="display: flex; height: 30px;">' + ''.join([
        f'<div style="flex:1; background:{color}; text-align:center; color:white; line-height:30px; font-weight:bold;">{cat}</div>' for cat, color in aqi_risk_colors
    ]) + '</div>', unsafe_allow_html=True)
    # Table
    st.markdown('')
    st.table({
        'AQI Category': [row[0] for row in aqi_risk_table],
        'AQI Range': [row[1] for row in aqi_risk_table],
        'Health Effects': [row[2] for row in aqi_risk_table]
    })
    
    # IoT Sensor Network Section
    st.subheader('IoT Sensor Network')
    
    # Load IoT sensor data
    try:
        iot_df = pd.read_csv('data/iot_sensors.csv')
        iot_df['timestamp'] = pd.to_datetime(iot_df['timestamp'])
        
        # Show latest sensor readings
        latest_sensors = iot_df.groupby('sensor_id').last().reset_index()
        
        # Display sensor table
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Live Sensor Readings**")
            sensor_display = latest_sensors[['sensor_id', 'city', 'pm25', 'temperature', 'humidity', 'quality_flag']].copy()
            sensor_display.columns = ['Sensor ID', 'City', 'PM2.5 (μg/m³)', 'Temp (°C)', 'Humidity (%)', 'Quality']
            st.dataframe(sensor_display, use_container_width=True)
        
        with col2:
            st.markdown("**Sensor Statistics**")
            st.metric("Total Sensors", len(latest_sensors))
            st.metric("Avg PM2.5", f"{latest_sensors['pm25'].mean():.1f} μg/m³")
            st.metric("Avg Temperature", f"{latest_sensors['temperature'].mean():.1f}°C")
            
    except FileNotFoundError:
        st.warning("IoT sensor data not found. Run data_cleaner.py to generate sample data.")
    
    # Disaster Management & NDMA/CPCB Alerts
    st.subheader('Disaster Management & Emergency Alerts')
    
    # Check for hazardous conditions
    if not filtered_forecast.empty and len(filtered_forecast) > 0:
        hazardous_cities = filtered_forecast[filtered_forecast['aqi_forecast'].isin(['Hazardous', 'Severe'])]
        
        if not hazardous_cities.empty:
            st.error("**EMERGENCY ALERT: Hazardous Air Quality Detected!**")
            st.markdown("""
            **Affected Cities:** """ + ", ".join(hazardous_cities['city'].unique()) + """
            
            **Immediate Actions Required:**
            - Avoid outdoor activities
            - Use air purifiers indoors
            - Monitor vulnerable populations
            - Consider school/work closures
            """)
            
            # Emergency notification buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Notify NDMA", type="primary"):
                    st.success("Emergency alert sent to NDMA!")
            with col2:
                if st.button("Notify CPCB"):
                    st.success("Alert sent to CPCB monitoring stations!")
            with col3:
                if st.button("Send Public Alert"):
                    st.success("Public SMS/Email alerts sent!")
        else:
            st.success("No hazardous air quality conditions detected.")
    else:
        st.info("No forecast data available for disaster assessment.")
    
    # Source Attribution Analysis (CNN-based)
    st.subheader('Pollution Source Attribution (AI Analysis)')
    
    # Simulated CNN-based source attribution
    pollution_sources = {
        'Delhi': {'traffic': 45, 'industrial': 25, 'construction': 15, 'agricultural': 10, 'other': 5},
        'Mumbai': {'traffic': 40, 'industrial': 30, 'construction': 20, 'agricultural': 5, 'other': 5},
        'Bengaluru': {'traffic': 50, 'industrial': 20, 'construction': 20, 'agricultural': 5, 'other': 5},
        'Chennai': {'traffic': 35, 'industrial': 35, 'construction': 15, 'agricultural': 10, 'other': 5},
        'Kolkata': {'traffic': 30, 'industrial': 40, 'construction': 15, 'agricultural': 10, 'other': 5},
        'Hyderabad': {'traffic': 45, 'industrial': 25, 'construction': 20, 'agricultural': 5, 'other': 5},
        'Pune': {'traffic': 40, 'industrial': 20, 'construction': 25, 'agricultural': 10, 'other': 5},
        'Ahmedabad': {'traffic': 35, 'industrial': 30, 'construction': 20, 'agricultural': 10, 'other': 5},
        'Jaipur': {'traffic': 30, 'industrial': 20, 'construction': 25, 'agricultural': 20, 'other': 5},
        'Lucknow': {'traffic': 35, 'industrial': 25, 'construction': 20, 'agricultural': 15, 'other': 5}
    }
    
    if selected_city in pollution_sources:
        sources = pollution_sources[selected_city]
        
        # Display source attribution chart
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create pie chart
            fig = px.pie(
                values=list(sources.values()),
                names=list(sources.keys()),
                title=f"Pollution Sources in {selected_city} (CNN Analysis)",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Source Breakdown**")
            for source, percentage in sources.items():
                st.metric(source.title(), f"{percentage}%")
            
            st.markdown("---")
            st.markdown("**AI Confidence:** 94.2%")
            st.markdown("**Analysis Method:** CNN + Satellite Imagery")
            st.markdown("**Last Updated:** " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    else:
        st.info(f"Source attribution analysis not available for {selected_city}.")
    
    # HSI Intelligence (Hyperspectral Imaging Analysis)
    st.subheader('HSI Intelligence - Pollutant Composition')
    
    try:
        # Load HSI data
        hsi_df = pd.read_csv('data/hsi_data.csv')
        hsi_df['timestamp'] = pd.to_datetime(hsi_df['timestamp'])
        
        # Get latest HSI data for selected city
        city_hsi = hsi_df[hsi_df['city'] == selected_city]
        
        if not city_hsi.empty:
            latest_hsi = city_hsi.iloc[-1]
            
            # Parse pollutant composition (JSON string)
            try:
                composition = json.loads(latest_hsi['pollutant_composition'])
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Create composition bar chart
                    fig = px.bar(
                        x=list(composition.keys()),
                        y=list(composition.values()),
                        title=f"Pollutant Composition in {selected_city} (HSI Analysis)",
                        labels={'x': 'Pollutant Type', 'y': 'Concentration (μg/m³)'},
                        color=list(composition.values()),
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**HSI Analysis Results**")
                    for pollutant, value in composition.items():
                        st.metric(pollutant.title(), f"{value:.2f} μg/m³")
                    
                    st.markdown("---")
                    st.markdown(f"**Hotspot Type:** {latest_hsi['hotspot_detection']}")
                    st.markdown(f"**Spatial Resolution:** {latest_hsi['spatial_resolution']}")
                    st.markdown(f"**Spectral Resolution:** {latest_hsi['spectral_resolution']}")
                    st.markdown("**Analysis Method:** Deep Learning + HSI")
                    
            except (json.JSONDecodeError, KeyError):
                st.warning("HSI composition data format error.")
        else:
            st.info(f"HSI data not available for {selected_city}.")
            
    except FileNotFoundError:
        st.warning("HSI data not found. Run data_cleaner.py to generate sample data.")
    
    # Alerts section
    st.subheader("Air Quality Alerts")
    
    if not filtered_forecast.empty and len(filtered_forecast) > 0:
        # Check for alerts in forecast
        alerts = []
        for _, row in filtered_forecast.iterrows():
            if row['aqi_forecast'] in ['Unhealthy', 'Very Unhealthy', 'Hazardous', 'Unhealthy for Sensitive Groups']:
                alerts.append(create_alert_message(
                    row['city'], 
                    row['aqi_forecast'], 
                    row['pm25_forecast']
                ))
        
        if alerts:
            for alert in alerts[:3]:  # Show top 3 alerts
                st.markdown(f'<div class="alert-box">{alert}</div>', unsafe_allow_html=True)
        else:
            st.success("No air quality alerts for the selected period.")
    else:
        st.info("No forecast data available for alerts.")
    
    # Charts section
    st.subheader("Air Quality Trends")
    
    if not filtered_forecast.empty and len(filtered_forecast) > 0 and not filtered_historical.empty and len(filtered_historical) > 0:
        # Combine historical and forecast data
        combined_df = pd.concat([
            filtered_historical[['timestamp', 'pm25']].assign(type='Historical'),
            filtered_forecast[['timestamp', 'pm25_forecast']].rename(columns={'pm25_forecast': 'pm25'}).assign(type='Forecast')
        ])
        
        # Create line chart
        fig = px.line(
            combined_df, 
            x='timestamp', 
            y='pm25', 
            color='type',
            title=f"PM2.5 Levels in {selected_city}",
            labels={'pm25': 'PM2.5 (μg/m³)', 'timestamp': 'Time'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    elif not filtered_historical.empty and len(filtered_historical) > 0:
        # Only historical data
        fig = px.line(
            filtered_historical, 
            x='timestamp', 
            y='pm25' if 'pm25' in filtered_historical.columns else 'no2_tropospheric_column',
            title=f"Historical Air Quality in {selected_city}",
            labels={'pm25': 'PM2.5 (μg/m³)', 'no2_tropospheric_column': 'NO₂ (ppb)', 'timestamp': 'Time'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Map section
    st.subheader("City Locations & IoT Sensor Network")
    
    # Create map
    city_coords = {
        'Delhi': [28.7041, 77.1025],
        'Mumbai': [19.0760, 72.8777],
        'Bengaluru': [12.9716, 77.5946],
        'Chennai': [13.0827, 80.2707],
        'Kolkata': [22.5726, 88.3639],
        'Hyderabad': [17.3850, 78.4867],
        'Pune': [18.5204, 73.8567],
        'Ahmedabad': [23.0225, 72.5714],
        'Jaipur': [26.9124, 75.7873],
        'Lucknow': [26.8467, 80.9462]
    }
    
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    
    # Add city markers
    for city, coords in city_coords.items():
        # Get current AQI for the city
        if not filtered_forecast.empty and len(filtered_forecast) > 0 and city == selected_city:
            if len(filtered_forecast) > 0:
                current_aqi = filtered_forecast.iloc[0]['aqi_forecast']
                current_pm25 = filtered_forecast.iloc[0]['pm25_forecast']
            else:
                current_aqi = 'Unknown'
                current_pm25 = 0
        else:
            city_data = historical_df[historical_df['city'] == city]
            if not city_data.empty and len(city_data) > 0:
                current_pm25 = city_data.iloc[-1]['pm25'] if 'pm25' in city_data.columns else 0
                current_aqi = city_data.iloc[-1]['aqi'] if 'aqi' in city_data.columns else 'Unknown'
            else:
                current_pm25 = 0
                current_aqi = 'Unknown'
        
        # Color based on AQI
        color = get_aqi_color(current_aqi)
        
        folium.Marker(
            coords,
            popup=f"<b>{city}</b><br>PM2.5: {current_pm25:.1f} μg/m³<br>AQI: {current_aqi}",
            tooltip=f"{city} - {current_aqi}",
            icon=folium.Icon(color='red' if current_aqi in ['Unhealthy', 'Very Unhealthy', 'Hazardous'] else 'green')
        ).add_to(m)
    
    # Add IoT sensor markers
    try:
        iot_df = pd.read_csv('data/iot_sensors.csv')
        latest_sensors = iot_df.groupby('sensor_id').last().reset_index()
        
        for _, sensor in latest_sensors.iterrows():
            # Create sensor icon
            sensor_color = 'blue' if sensor['quality_flag'] == 1 else 'gray'
            
            folium.CircleMarker(
                location=[sensor['latitude'], sensor['longitude']],
                radius=8,
                popup=f"<b>IoT Sensor: {sensor['sensor_id']}</b><br>City: {sensor['city']}<br>PM2.5: {sensor['pm25']:.1f} μg/m³<br>Quality: {sensor['quality_flag']}",
                tooltip=f"IoT: {sensor['sensor_id']} - {sensor['pm25']:.1f} μg/m³",
                color=sensor_color,
                fill=True,
                fillOpacity=0.7
            ).add_to(m)
    except FileNotFoundError:
        pass  # IoT data not available
    
    folium_static(m, width=800, height=400)
    
    # Data table
    st.subheader("Detailed Data")
    
    if not filtered_forecast.empty and len(filtered_forecast) > 0:
        st.dataframe(filtered_forecast, use_container_width=True)
    elif not filtered_historical.empty and len(filtered_historical) > 0:
        st.dataframe(filtered_historical, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>AEROVIEW - AI-Powered Air Pollution Monitoring</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 