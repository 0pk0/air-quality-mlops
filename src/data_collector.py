import requests
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
from numpy.distutils.system_info import dfftw_info


class AirQualityCollector:
    def __init__(self):
        load_dotenv()
        self.cities = [
            {'name': 'London', 'lat': 51.5074, 'lon':-0.1278},
            {'name': 'Stirchley, UK', 'lat': 52.434, 'lon':-1.9248},
            {'name': 'Delhi', 'lat': 28.6139, 'lon':77.2090},
            {'name': 'Tokyo', 'lat': 35.6762, 'lon': 139.6503}
        ]

    def collect_sample_data(self):
        data = []
        for city in self.cities:
            sample_record = {
                'city': city['name'],
                'timestamp': datetime.now(),
                'pm2_5': 25.5,
                'pm10': 45.5,
                'temperature': 22.1,
                'humidity': 70
            }
            data.append(sample_record)

            df = pd.DataFrame(data)
            os.makedirs('data', exist_ok=True)
            df.to_csv('data/sample_air_quality.csv', index=False)
            print('Saved')
            print (df)
            return df

if __name__ == '__main__':
    collector = AirQualityCollector()
    data = collector.collect_sample_data()

