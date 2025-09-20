import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, data_dir:str = 'data'):
        self.data_dir = data_dir
        self.cities = [
            {'name': 'London', 'lat': 51.5074, 'lon': -0.1278, 'baseline_pm25': 15},
            {'name': 'Tokyo', 'lat': 35.6762, 'lon': 139.6503, 'baseline_pm25': 25},
            {'name': 'Delhi', 'lat': 28.6139, 'lon': 77.2090, 'baseline_pm25': 45}
        ]
        self.ensure_data_dir()

    def ensure_data_dir(self):
        os.makedirs(self.data_dir, exist_ok=True)
        logger.info(f"Data directory {self.data_dir} created")

    def generate_synthetic_data(self, days: int=30) -> pd.DataFrame:
        logger.info(f"Generating {days} days synthetic data")

        data_records = []
        base_date = datetime.now() + timedelta(days=days)


        for city in self.cities:
            for day in range(days):
                current_date = base_date + timedelta(days=day)

                #For realisitc patterns
                seasonal_effect = np.sin(day * 2 * np.pi / 365) * 8  # Yearly cycle
                weekly_effect = np.sin(day * 2 * np.pi / 7) * 3  # Weekly cycle
                random_noise = np.random.normal(0, 3)

                pm25 = city['baseline_pm25'] + seasonal_effect + weekly_effect + random_noise
                pm25 = max(5.0, pm25)

                temp_base = 20 + seasonal_effect * 0.8
                temperature = temp_base + np.random.normal(0, 3)

                humidity = 60 + np.random.normal(0, 12)
                humidity = np.clip(humidity, 30, 90)

                wind_speed = abs(np.random.normal(8,3))
                
                # Add atmospheric pressure
                pressure = 1013.25 + np.random.normal(0, 10)  # Standard pressure with variation

                record = {
                    'city': city['name'],
                    'date': current_date.strftime('%Y-%m-%d'),
                    'timestamp': current_date,
                    'pm2_5': round(pm25, 2),
                    'temperature': round(temperature, 1),
                    'pressure': round(pressure, 2),
                    'humidity': round(humidity, 1),
                    'wind_speed': round(wind_speed, 1),
                    'latitude': city['lat'],
                    'longitude': city['lon'],
                    'day_of_week': current_date.weekday(),
                    'month': current_date.month,
                    'day_of_year': current_date.timetuple().tm_yday
                }

                data_records.append(record)

        df = pd.DataFrame(data_records)
        df = df.sort_values(['city', 'date']).reset_index(drop=True)

        logger.info(f"Generated {len(df)} for {len(self.cities)} cities")
        return df

    def save_data(self, df: pd.DataFrame, filename: str = 'air_quality.csv'):
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
        return filepath

    def load_data(self, filename: str = 'air_quality.csv') -> Optional[pd.DataFrame]:
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            logger.warning(f"filenot found at {filepath}")
            return None

        try:
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            logger.info(f"Loaded {len(df)} records from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            return None

    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        summary = {
            'total_records': len(df),
            'cities': df['city'].unique().tolist(),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max()
            },
            'pm25_stats': {
                'mean': round(df['pm2_5'].mean(), 2),
                'min': round(df['pm2_5'].min(), 2),
                'max': round(df['pm2_5'].max(), 2),
                'std': round(df['pm2_5'].std(), 2)
            },
            'city_stats': df.groupby('city')['pm2_5'].agg(['mean', 'min', 'max']).round(2).to_dict()
        }

        return summary

def main():
    print("Test")
    dm = DataManager()
    df = dm.generate_synthetic_data(30)
    print({df.shape})
    filepath = dm.save_data(df)
    print(filepath)
    df_loaded = dm.load_data()
    print({df_loaded.shape})
    summary = dm.get_data_summary(df_loaded)
    print(f"\n📊 Data Summary:")
    print(f"   Total records: {summary['total_records']}")
    print(f"   Cities: {summary['cities']}")
    print(f"   Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"   PM2.5 range: {summary['pm25_stats']['min']} - {summary['pm25_stats']['max']} µg/m³")


if __name__ == "__main__":
    main()

