from apscheduler.schedulers.background import BackgroundScheduler
from model_updater import ModelUpdater
from data_fetcher import DataFetcher
from data_preprocessor import DataPreprocessor
from macroeconomic_data import MacroeconomicData
from config import Config

class RetrainScheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.model_updater = ModelUpdater()
        self.data_fetcher = DataFetcher()
        self.data_preprocessor = DataPreprocessor()
        self.macro_data = MacroeconomicData(Config.FRED_API_KEY)

    def retrain_model(self):
        stock_data = self.data_fetcher.fetch_historical_stock_data(Config.STOCK_TICKER, Config.START_DATE, Config.END_DATE)
        unemployment_rate = self.macro_data.fetch_historical_data('UNRATE', Config.START_DATE, Config.END_DATE)
        interest_rate = self.macro_data.fetch_historical_data('FEDFUNDS', Config.START_DATE, Config.END_DATE)
        inflation_rate = self.macro_data.fetch_historical_data('CPIAUCSL', Config.START_DATE, Config.END_DATE)
        
        combined_data = stock_data.copy()
        combined_data['UnemploymentRate'] = unemployment_rate
        combined_data['InterestRate'] = interest_rate
        combined_data['InflationRate'] = inflation_rate

        scaled_data = self.data_preprocessor.scale_data(combined_data)

        X, y = self.data_preprocessor.create_dataset(scaled_data, Config.TIME_STEP)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
        self.model_updater.update_model(X, y)

    def start(self):
        self.scheduler.add_job(self.retrain_model, 'interval', hours=24)
        self.scheduler.start()
