from data_fetcher import DataFetcher
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from macroeconomic_data import MacroeconomicData
from config import Config

data_fetcher = DataFetcher()
data_preprocessor = DataPreprocessor()
model_trainer = ModelTrainer()
macro_data = MacroeconomicData(Config.FRED_API_KEY)

# Fetch historical data
ticker = Config.STOCK_TICKER
stock_data = data_fetcher.fetch_historical_stock_data(ticker, Config.START_DATE, Config.END_DATE)
stock_data_index = stock_data.index

unemployment_rate = macro_data.fetch_historical_data('UNRATE', Config.START_DATE, Config.END_DATE, stock_data_index)
interest_rate = macro_data.fetch_historical_data('FEDFUNDS', Config.START_DATE, Config.END_DATE, stock_data_index)
inflation_rate = macro_data.fetch_historical_data('CPIAUCSL', Config.START_DATE, Config.END_DATE, stock_data_index)

# Combine data
combined_data = stock_data.copy()
combined_data['UnemploymentRate'] = unemployment_rate
combined_data['InterestRate'] = interest_rate
combined_data['InflationRate'] = inflation_rate

# Scale data
scaled_data = data_preprocessor.scale_data(combined_data)

# Prepare data for training
time_step = Config.TIME_STEP
X, y = data_preprocessor.create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

# Build and train model
model = model_trainer.build_model((X.shape[1], X.shape[2]))
model_trainer.train_model(model, X, y)

print("Model trained and saved as 'stock_predictor.h5'")
