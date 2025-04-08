import sys
import os
from datetime import datetime, timedelta
from src.Stock_Movement_Predicition.logging import logger
from src.Stock_Movement_Predicition.exception import StockMovingPredicitionException
from src.Stock_Movement_Predicition.components.data_ingestion import DataIngestion
from src.Stock_Movement_Predicition.components.data_preprocessing import FeatureEngineering
from dotenv import load_dotenv

load_dotenv()


finn_api_key = os.getenv('finn_api_key')
vantage_api_key = os.getenv('vantage_api_key')
symbol = 'AAPL'
months=24
if __name__ == '__main__':
    logger.info("Data Ingestion has Started")
    data_ingestion = DataIngestion()
    full_data = data_ingestion.initiate_data_ingestion(
        symbol=symbol,
        months=months
    )
    logger.info("Data Ingestion Completed Successfully")

    logger.info("Data Preprocessing has Started Successfully")

    data_preprocessing = FeatureEngineering()
    final_data = data_preprocessing.initiate_feature_engineering(full_data)
    logger.info("Data Preprocessing Completed Successfully")




