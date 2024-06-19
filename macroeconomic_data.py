from fredapi import Fred
import requests
import pandas as pd

class MacroeconomicData:
    def __init__(self, api_key):
        self.fred = Fred(api_key=api_key)

    def fetch_historical_data(self, series_id, start_date, end_date, stock_data_index):
        url = f'https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={self.fred.api_key}&file_type=json&observation_start={start_date}&observation_end={end_date}'
        response = requests.get(url, verify=False)  # Disable SSL verification
        data = response.json()['observations']
        dates = [item['date'] for item in data]
        values = [float(item['value']) for item in data]
        series = pd.Series(values, index=pd.to_datetime(dates))
        # Ensure the timezone matches
        series.index = series.index.tz_localize('UTC').tz_convert(stock_data_index.tz)
        return series.reindex(stock_data_index, method='ffill')
