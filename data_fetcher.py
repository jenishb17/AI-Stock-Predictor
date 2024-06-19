import yfinance as yf
import pandas as pd

class DataFetcher:
    def fetch_historical_stock_data(self, ticker, start_date, end_date):
        return yf.download(ticker, start=start_date, end=end_date)
    
    def fetch_real_time_stock_data(self, ticker):
        return yf.download(ticker, period='1d', interval='1m')
