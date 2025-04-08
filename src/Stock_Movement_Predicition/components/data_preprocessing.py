import os
import sys
import pandas as pd
import numpy as np
from datetime import timedelta
import ta  
from src.Stock_Movement_Predicition.logging import logger
from src.Stock_Movement_Predicition.exception import StockMovingPredicitionException

class FeatureEngineering:
    def __init__(self):
        try:
            logger.info("FeatureEngineering initialized.")
        except Exception as e:
            raise StockMovingPredicitionException(e, sys)

    def add_price_based_features(self, df):
        df['lag_1_close'] = df['Close'].shift(1)
        df['lag_1_open'] = df['Open'].shift(1)
        df['lag_1_high'] = df['High'].shift(1)
        df['lag_1_low'] = df['Low'].shift(1)
        df['lag_1_volume'] = df['Volume'].shift(1)

        df['lag_1_return'] = (df['lag_1_close'] - df['lag_1_open']) / df['lag_1_open']
        df['lag_2_return'] = df['lag_1_return'].shift(1)
        df['lag_3_return'] = df['lag_1_return'].shift(2)

        df['cumulative_return_3'] = df['lag_1_return'].rolling(3).sum()

        df['SMA_5'] = df['lag_1_close'].rolling(window=5).mean()
        df['SMA_10'] = df['lag_1_close'].rolling(window=10).mean()
        df['EMA_10'] = df['lag_1_close'].ewm(span=10, adjust=False).mean()
        df['EMA_20'] = df['lag_1_close'].ewm(span=20, adjust=False).mean()

        df['MACD'] = ta.trend.macd(df['lag_1_close'])
        df['RSI'] = ta.momentum.rsi(df['lag_1_close'])

        df['bollinger_h'] = ta.volatility.bollinger_hband(df['lag_1_close'])
        df['bollinger_l'] = ta.volatility.bollinger_lband(df['lag_1_close'])

        df['volatility'] = df['lag_1_close'].rolling(window=5).std()
        df['rolling_std_5'] = df['lag_1_close'].rolling(window=5).std()
        df['close_to_open_ratio'] = df['lag_1_close'] / df['lag_1_open']
        df['high_to_low_ratio'] = df['lag_1_high'] / df['lag_1_low']

        return df

    def add_volume_features(self, df):
        df['volume_change'] = df['lag_1_volume'].pct_change(fill_method=None)
        df['volume_SMA_5'] = df['lag_1_volume'].rolling(window=5).mean()
        df['lag_2_volume'] = df['lag_1_volume'].shift(1)
        return df

    def add_sentiment_features(self, df):
        if 'sentiment_score' in df.columns:
            df['sentiment_momentum'] = df['sentiment_score'].diff()
            df['rolling_sentiment_3'] = df['sentiment_score'].rolling(3).mean()

        if 'text' in df.columns:
            df['news_count'] = df['text'].apply(lambda x: 1 if pd.notnull(x) and x.strip() != '' else 0)
        return df

    def add_temporal_features(self, df):
        df['day_of_week'] = pd.to_datetime(df['Date']).dt.dayofweek
        df['month'] = pd.to_datetime(df['Date']).dt.month
        df['quarter'] = pd.to_datetime(df['Date']).dt.quarter
        df['is_month_end'] = pd.to_datetime(df['Date']).dt.is_month_end.astype(int)
        return df

    def add_target_label(self, df):
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        return df

    def initiate_feature_engineering(self, df):
        try:
            df = self.add_price_based_features(df)
            df = self.add_volume_features(df)
            df = self.add_sentiment_features(df)
            df = self.add_temporal_features(df)
            df = self.add_target_label(df)
            os.makedirs("data", exist_ok=True)
            df.to_csv("data/AAPL_final_dataset.csv", index=False)
            logger.info("Saved feature engineered dataset to data/AAPL_final_dataset.csv")
            return df
        except Exception as e:
            raise StockMovingPredicitionException(e, sys)
