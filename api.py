from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

import sentiment_engine as se
import twitter_fetcher as tf

app = FastAPI(title="Sentiment Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class AnalyzeRequest(BaseModel):
    text: str

class TweetResponse(BaseModel):
    id: str
    text: str
    created_at: str
    author_id: str
    public_metrics: dict
    sentiment: Optional[dict] = None

@app.post("/api/analyze")
def analyze_text(req: AnalyzeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    pipeline_steps = se.run_pipeline(req.text)
    clean_text = se.get_clean_text(req.text)
    
    # Sentence-level sentiment on ORIGINAL text (VADER works best on raw text)
    s_sentiment = se.sentence_sentiment(req.text)
    
    # Word-level with mathematically derived metrics (on clean text)
    w_metrics = se.word_sentiment_with_metrics(clean_text)
    w_sentiment = w_metrics["words"]
    a_sentiment = se.aspect_sentiment(clean_text)
    
    return {
        "pipeline_steps": pipeline_steps,
        "clean_text": clean_text,
        "sentence_sentiment": s_sentiment,
        "word_sentiment": w_sentiment,
        "aspect_sentiment": a_sentiment,
        "metrics": {
            "mean_compound": w_metrics["mean_compound"],
            "distribution": w_metrics["distribution"],
            "category": w_metrics["category"],
        }
    }

@app.get("/api/tweet")
def fetch_tweet(url: str):
    try:
        tweet_data = tf.fetch_tweet_by_url(url)
        # also run sentiment on it
        text = tweet_data.get("text", "")
        clean_text = se.get_clean_text(text)
        sentiment = se.sentence_sentiment(clean_text)
        
        return {
            "tweet": tweet_data,
            "sentiment": sentiment
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/hashtag")
def fetch_hashtag(hashtag: str, max_results: int = 10):
    try:
        tweets = tf.fetch_tweets_by_hashtag(hashtag, max_results)
        results = []
        for t in tweets:
            text = t.get("text", "")
            clean_text = se.get_clean_text(text)
            sentiment = se.sentence_sentiment(clean_text)
            results.append({
                "tweet": t,
                "sentiment": sentiment
            })
        return {"tweets": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Twitiment API is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

