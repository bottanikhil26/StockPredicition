import os
import sys
import json
import pandas as pd
import numpy as np
from src.Stock_Movement_Predicition.exception import StockMovingPredicitionException
from src.Stock_Movement_Predicition.logging import logger
from dateutil.relativedelta import relativedelta
import requests
from datetime import datetime, timedelta
import os
import time




from dotenv import load_dotenv
load_dotenv()

finn_api_key = os.getenv('finn_api_key')
vantage_api_key = os.getenv('vantage_api_key')

class DataIngestion():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise StockMovingPredicitionException(e,sys)
        

    def fetch_finnhub_news_daily(self, symbol: str, finn_api_key: str, start_date: datetime, end_date: datetime):
        try:
            base_url = "https://finnhub.io/api/v1/company-news"
            all_articles = []

            current_date = start_date
            while current_date <= end_date:
                print(f"Fetching news for {current_date}...")

                params = {
                    "symbol": symbol,
                    "from": current_date.isoformat(),
                    "to": current_date.isoformat(),
                    "token": finn_api_key
                }

                response = requests.get(base_url, params=params)
                data = response.json()

                if isinstance(data, list) and data:
                    for article in data:
                        article["fetched_date"] = current_date.isoformat()
                    all_articles.extend(data)

                time.sleep(1)  # Respect rate limits
                current_date += timedelta(days=1)

            df = pd.DataFrame(all_articles)

            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], unit='s', errors='coerce')
                df = df.dropna(subset=['datetime'])
            return df
        
        except Exception as e:
            return StockMovingPredicitionException(e, sys)
        



    def fetch_alpha_vantage_stock_data(self, symbol: str, start_date: datetime, end_date: datetime, vantage_api_key: str):
        try:
            # Alpha Vantage API endpoint for daily stock data (time series)
            base_url = "https://www.alphavantage.co/query"
            function = "TIME_SERIES_DAILY"
            
            params = {
                "function": function,
                "symbol": symbol,
                "apikey": vantage_api_key,
                "outputsize": "full"  # "full" for all available data
            }

            response = requests.get(base_url, params=params)
            data = response.json()

            # Check if data is available
            if "Time Series (Daily)" in data:
                time_series = data["Time Series (Daily)"]
                df = pd.DataFrame(time_series).T  # Transpose to have dates as rows
                df.reset_index(inplace=True)
                df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

                # Convert types
                df['Date'] = pd.to_datetime(df['Date'])
                df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

                # Filter data for the last `months` months
                df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

                # Sort by date descending
                df = df.sort_values(by='Date', ascending=False).reset_index(drop=True)

                return df
            
        except Exception as e:
            raise StockMovingPredicitionException(e)
    
    def save_news_to_data(self,df: pd.DataFrame, symbol: str, months: int):
        try:
            folder = "data"
            os.makedirs(folder, exist_ok=True)
            file_path = os.path.join(folder, f"{symbol}_finnhub_daily_news_{months}months.csv")
            df.to_csv(file_path, index=False)
            print(f"Saved daily news for {months} months to {file_path}")

        except Exception as e:
            return StockMovingPredicitionException(e,sys)
        
    def save_stock_data_to_csv(self, df: pd.DataFrame, symbol: str, months: int):
        try:
            folder = "data"
            os.makedirs(folder, exist_ok=True)
            file_path = os.path.join(folder, f"{symbol}_stock_data_{months}months.csv")
            df.to_csv(file_path, index=False)
            print(f"Saved stock data for {months} months to {file_path}")
        except Exception as e:
            raise StockMovingPredicitionException(e,sys)
        
    
    
    
        

    def initiate_data_ingestion(self, symbol: str, months: int, vantage_api_key: str, finn_api_key: str):
        try:
            end_date = datetime.today().date()
            start_date = end_date - timedelta(days=months * 30)

        # Fetch and Save Stock Data
            stock_data_df = self.fetch_alpha_vantage_stock_data(symbol, start_date, end_date, vantage_api_key)
            self.save_stock_data_to_csv(stock_data_df, symbol, months)

        # Fetch and Save News Data
            news_df = self.fetch_finnhub_news_daily(symbol, finn_api_key, start_date, end_date)
            self.save_news_to_data(news_df, symbol, months)


        except Exception as e:
            raise StockMovingPredicitionException(e, sys)
    
        
    
if __name__ == '__main__':
    ingestor = DataIngestion()

# Run the ingestion pipeline
    ingestor.initiate_data_ingestion(
        symbol="AAPL",          # <-- You can change the stock symbol here
        months=24,               # <-- Last 2 year data but we are getting news data for only 1 year
        vantage_api_key=vantage_api_key,
        finn_api_key=finn_api_key
    )

