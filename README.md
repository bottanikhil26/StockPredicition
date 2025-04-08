
# Stock Price Movement Prediction

This project is an end-to-end machine learning pipeline that predicts the **next day's stock price movement (Up/Down)** for `AAPL`, based on historical stock data and sentiment analysis of news headlines. It includes data ingestion, preprocessing, feature engineering, model training, and a REST API for serving predictions.

---

## 📦 Project Structure

```
.
├── data/                          # Stores raw and processed CSV data
│   ├── AAPL_raw_stock.csv
│   ├── AAPL_news_data.csv
│   └── AAPL_final_dataset.csv
│
├── src/
│   └── Stock_Movement_Prediction/
│       ├── etl.py                # Pulls stock and news data
│       ├── data_ingestion.py     # Merges data and performs sentiment analysis
│       ├── transformation.py     # Performs feature engineering
│       ├── model_trainer.py      # Trains and saves LightGBM model
│       ├── prediction.py         # Loads model and makes predictions
│       ├── api.py                # FastAPI app with /predict endpoint
│       ├── logging.py            # Custom logging
│       └── exception.py          # Custom exception handling
│
├── Dockerfile                    # Containerization
├── requirements.txt              # Required packages
└── README.md                     # You're here
```

---

# Step 1: ETL Pipeline – Data Collection and Storage

Stock Data: Collected using the Alpha Vantage API (or similar), storing historical Open, High, Low, Close, and Volume values for the ticker AAPL.
News Data: Fetched from the Finnhub API, providing headline and summary text related to the stock.
Storage: Both stock and news data are saved in the data/ folder.
Files generated:

data/AAPL_stock_data.csv
data/AAPL_news_data.csv
# Step 2: Data Ingestion and Sentiment Analysis

Text Processing: The news data (headline + summary) is passed through the pre-trained FinBERT model to compute a sentiment score.
Merging: Sentiment scores are merged with stock data based on the date column.
Cleaned Dataset: The merged dataset includes daily OHLCV data and corresponding sentiment.
Output:

data/AAPL_full_dataset.csv — merged dataset with stock and sentiment data.
Step 3: Feature Engineering and Prediction

Feature Engineering
Using the FeatureEngineering class in data_preprocessing.py, the following features are added based on the lagged and historical stock/sentiment data:

Price-Based Features:
lag_1_close, lag_1_open, lag_1_high, lag_1_low, lag_1_volume
lag_1_return: Previous day return
cumulative_return_3: Sum of 3-day past returns
SMA_5, SMA_10: Simple moving averages
EMA_10, EMA_20: Exponential moving averages
MACD, RSI: Momentum indicators
bollinger_h, bollinger_l: Bollinger bands
volatility: Rolling standard deviation
Volume Features:
volume_change: Day-over-day percentage change
volume_SMA_5: 5-day average volume
Sentiment Features:
sentiment_momentum: Day-to-day change in sentiment
news_count: Presence/absence of news for the day
Temporal Features:
day_of_week, month: Encodes seasonal patterns
Target Variable:
target: Binary label indicating whether the next day's closing price is higher (1) or lower (0)
Output:

data/AAPL_final_dataset.csv
Prediction API
The /predict endpoint in FastAPI:

Accepts a stock ticker (e.g., AAPL) and date range
Checks for missing dates and adds placeholders
Applies feature engineering
Loads the trained LightGBM model from models/AAPL_lightgbm.pkl
Predicts movement or shows actual target if already available

## 🧠 Model Training

- Approach: Machine learning (LightGBM) due to limited data availability.
- Data Merging: Merged stock data and text data (from the past year).
- Data Limitation: After preprocessing and feature engineering, only 230    rows of data remained.
- Model Choice: Opted for LightGBM instead of LSTM, as LSTM typically requires a larger dataset.
- Hyperparameter Tuning: The LightGBM model has been fine-tuned using Optuna.
- Performance Evaluation: Evaluated using multiple metrics:
- Classification accuracy
- Precision
- Recall
- F1 score
- AUC curve

---

##  API Exposure (app.py)

A FastAPI app exposes a RESTful prediction endpoint.

### Endpoint: `/predict`

#### Request:
```json
{
  "ticker": "AAPL",
  "date": "2023-08-24"
}
```

#### Response:
```json
{
  "prediction": "UP",
  "confidence": 0.84
}
```

- Checks if the requested date is available in data.
- If not, fetches missing data, processes it, appends to dataset, and makes a prediction.

---

## 🐳 Docker Instructions

### Build the Image:
```bash
docker build -t stock-movement-api .
```

### Run the Container:
```bash
docker run -p 8000:8000 stock-movement-api
```

### Test the API:
Visit [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI.

---

## ⚙️ Setup Instructions

### Local Setup

```bash
git clone https://github.com/bottanikhil26/StockPredicition.git

for running the docker file
docker build -t condor-stock-api .
docker run -p 8000:8000 condor-stock-api

swagger ui - http://localhost:8000/docs (enter this your chromaor sarfari)


uvicorn app:app --reload (for checking the fast api)
```

---

## 📌 Assumptions
- We are running the etl code daily by using Airflow
- FinBERT sentiment score influences short-term movement.
- We are are tracking model performs weekly by Ml Flow 
- Prediction is binary classification: next-day `UP` or `DOWN`.
- Only data for ticker `AAPL` is used.

---

## ✅ Evaluation Summary

- ✅ Modular ETL & Feature Engineering
- ✅ FinBERT-based sentiment integration
- ✅ Tuned LightGBM classifier
- ✅ Real-time prediction with dynamic data fetching
- ✅ Dockerized for portability
- ✅ RESTful FastAPI interface
