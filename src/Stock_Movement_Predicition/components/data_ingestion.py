import pandas as pd
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.Stock_Movement_Predicition.exception import StockMovingPredicitionException

class DataIngestion:
    def __init__(self):
        try:
            # Load model and tokenizer only once to avoid redundant calls
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.label_mapping = {"negative": -1, "neutral": 0, "positive": 1}
            self.id2label = self.model.config.id2label
        except Exception as e:
            raise StockMovingPredicitionException(f"Error loading tokenizer/model: {str(e)}", sys)

    def perform_sentiment_analysis(self, texts):
        try:
            sentiment_labels = []
            sentiment_scores = []
            
            # Process texts in batches for efficiency
            batch_size = 32  # Adjust this based on your available memory
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model(**inputs)

                logits = outputs.logits
                probs = F.softmax(logits, dim=1)
                scores, preds = torch.max(probs, dim=1)

                # Collect results for batch
                sentiment_labels.extend([self.id2label[pred.item()].lower() for pred in preds])
                sentiment_scores.extend(scores.tolist())

            sentiment_numeric = [self.label_mapping.get(label, 0) for label in sentiment_labels]
            return sentiment_scores, sentiment_numeric
        except Exception as e:
            raise StockMovingPredicitionException(f"Error during sentiment analysis: {str(e)}", sys)

    def initiate_data_ingestion(self, symbol: str,months: int):
        try:
            # File paths
            stock_file_path = f"data/{symbol}_stock_data_{months}months.csv"
            news_file_path = f"data/{symbol}_finnhub_daily_news_{months}months.csv"

            # Read data
            stock_df = pd.read_csv(stock_file_path)
            news_df = pd.read_csv(news_file_path)

            # Combine headline + summary
            news_df["text"] = news_df["headline"].fillna('') + ". " + news_df["summary"].fillna('')
            news_df["fetched_date"] = pd.to_datetime(news_df["fetched_date"]).dt.date

            # Sentiment analysis
            sentiment_scores, sentiment_labels = self.perform_sentiment_analysis(news_df["text"].tolist())
            news_df["sentiment_score"] = sentiment_scores
            news_df["sentiment_label"] = sentiment_labels
            news_df["is_positive"] = (news_df["sentiment_label"] == 1).astype(int)

            # Aggregate sentiment by date
            sentiment_df = news_df.groupby("fetched_date").agg(
                sentiment_score=("sentiment_score", "mean"),
                sentiment_label=("sentiment_label", lambda x: x.mode().iloc[0] if not x.mode().empty else 0),
                positive_ratio=("is_positive", "mean"),
                news_count=("is_positive", "count")
            ).reset_index()

            stock_df["Date"] = pd.to_datetime(stock_df["Date"]).dt.date

            # Merge datasets
            full_df = pd.merge(stock_df, sentiment_df, left_on="Date", right_on="fetched_date", how="inner")
            full_df.drop(columns=["fetched_date"], inplace=True)
            full_df.sort_values(by="Date", inplace=True)
            full_df.reset_index(drop=True, inplace=True)    

            # Handle missing values
            full_df["sentiment_score"] = full_df["sentiment_score"].fillna(0.0)
            full_df["sentiment_label"] = full_df["sentiment_label"].fillna(0).astype(int)

            # Save the merged dataset
            os.makedirs("data", exist_ok=True)
            full_path = f"data/{symbol}_full_dataset.csv"
            full_df.to_csv(full_path, index=False)

            print(f"Full dataset with sentiment saved at '{full_path}'")
            return full_df

        except Exception as e:
            raise StockMovingPredicitionException(f"Error during data ingestion: {str(e)}", sys)






        
        
