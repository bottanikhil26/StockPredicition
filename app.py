from fastapi import FastAPI, Query, HTTPException
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from typing import Optional
from uvicorn import run as app_run
import pandas as pd
import joblib
import os

from src.Stock_Movement_Predicition.components.data_preprocessing import FeatureEngineering

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.get("/", tags=["home"])
def home():
    return RedirectResponse(url="/docs")

@app.get("/predict")
async def predict(
    ticker: str = Query(..., description="Stock ticker symbol, e.g., AAPL"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    symbol = ticker.upper()
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        # Load existing full dataset
        full_path = f"data/{symbol}_full_dataset.csv"
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="Full dataset not found.")
        df = pd.read_csv(full_path, parse_dates=["Date"])
        df["Date"] = pd.to_datetime(df["Date"]).dt.date

        # Filter requested date range
        requested_dates = set(pd.date_range(start, end).date)
        available_dates = set(df["Date"].unique())
        missing_dates = sorted(list(requested_dates - available_dates))

        if missing_dates:
            print(f"Missing dates found: {missing_dates}")
            # Add placeholder rows
            for date in missing_dates:
                df = pd.concat([df, pd.DataFrame([{
                    "Date": date,
                    "Open": None, "High": None, "Low": None, "Close": None, "Volume": None,
                    "sentiment_score": None, "text": None
                }])], ignore_index=True)

            df = df.drop_duplicates(subset=["Date"]).sort_values("Date")
            df.to_csv(full_path, index=False)
            print(f"Updated full dataset with missing dates.")

        # Feature Engineering
        fe = FeatureEngineering()
        df_fe = fe.initiate_feature_engineering(df.copy())
        df_fe["Date"] = pd.to_datetime(df_fe["Date"]).dt.date

        # Filter engineered data to requested range
        df_final = df_fe[(df_fe["Date"] >= start) & (df_fe["Date"] <= end)]

        if df_final.empty:
            raise HTTPException(status_code=400, detail="No feature-engineered data in range")

        # Load model
        model_path = f"models/{symbol}_lightgbm_6.pkl"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found.")
        model = joblib.load(model_path)

        # Get top 15 features by importance
        if hasattr(model, "feature_importances_") and hasattr(model, "feature_name_"):
            importance_df = pd.DataFrame({
                "feature": model.feature_name_,
                "importance": model.feature_importances_
            }).sort_values(by="importance", ascending=False).head(15)
            top_features = importance_df.to_dict(orient="records")
        else:
            top_features = []

        # Drop columns not used in prediction
        drop_cols = ["Date", "target", "Open", "High", "Low", "Close", "Volume", "text", "sentiment_score"]
        X = df_final.drop(columns=drop_cols, errors="ignore")

        # Predict or use actual target
        predictions = []
        for i, row in df_final.iterrows():
            date = row["Date"]

            if not pd.isna(row.get("target")):
                label = int(row["target"])
                pred = "UP" if label == 1 else "DOWN"
                source = "predicted"
            else:
                row_input = row.drop(labels=drop_cols, errors="ignore").values.reshape(1, -1)

                if row_input is not None and not pd.isna(row_input).any():
                    label = model.predict(row_input)[0]
                    pred = "UP" if label == 1 else "DOWN"
                    source = "predicted"
                else:
                    pred = "N/A"
                    source = "insufficient_data"

            predictions.append({
                "date": str(date),
                "prediction": pred,
                "source": source
            })

        return {
            "symbol": symbol,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "predictions": predictions,
            "top_15_features": top_features
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__=="__main__":
    app_run(app,host="0.0.0.0",port=8000)
