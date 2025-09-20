import requests
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv


class AirQualityCollector:
    def __init__(self):
        load_dotenv()
        # We'll use free APIs first
        self.cities = [
            {'name': 'London', 'lat': 51.5074, 'lon': -0.1278},
            {'name': 'Tokyo', 'lat': 35.6762, 'lon': 139.6503},
            {'name': 'Delhi', 'lat': 28.6139, 'lon': 77.2090}
        ]

    def collect_sample_data(self):
        """Collect sample air quality data"""
        print("air quality data collection...")

        # For now, create sample data structure
        data = []
        for city in self.cities:
            sample_record = {
                'city': city['name'],
                'timestamp': datetime.now(),
                'pm2_5': 25.5,  # We'll get real data next
                'pm10': 45.2,
                'temperature': 22.1,
                'humidity': 65
            }
            data.append(sample_record)

        df = pd.DataFrame(data)
        df.to_csv('data/sample_air_quality.csv', index=False)
        print("✅ Sample data saved to data/sample_air_quality.csv")
        return df


if __name__ == "__main__":
    collector = AirQualityCollector()
    df = collector.collect_sample_data()
    print(df)
