"""
Twitter Fetcher — Logic for interacting with the Twitter/X API using Tweepy.
"""
import tweepy
import re
import streamlit as st

import os

# Load .env file variables manually if present in workspace
_env_file = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_file):
    with open(_env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

# Fetch keys from environment variables
BEARER_TOKEN = os.getenv("BEARER_TOKEN")
CONSUMER_KEY = os.getenv("CONSUMER_KEY")
CONSUMER_SECRET = os.getenv("CONSUMER_SECRET")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET")


def get_client():
    """Initialise and return a Tweepy Client."""
    # Using Bearer Token is the simplest way for App-Only v2 data fetching
    return tweepy.Client(
        bearer_token=BEARER_TOKEN,
        consumer_key=CONSUMER_KEY,
        consumer_secret=CONSUMER_SECRET,
        access_token=ACCESS_TOKEN,
        access_token_secret=ACCESS_TOKEN_SECRET
    )

@st.cache_data(show_spinner=False, ttl=600)
def fetch_tweet_by_url(url_or_id: str) -> dict:
    """
    Fetch a single tweet's data by its URL or numeric ID.
    """
    # Extract numeric ID if a full URL is provided
    tweet_id = url_or_id
    if "status/" in url_or_id:
        tweet_id = url_or_id.split("status/")[-1].split("?")[0]
    
    client = get_client()
    response = client.get_tweet(
        tweet_id,
        tweet_fields=["created_at", "public_metrics", "text", "author_id", "lang"]
    )
    
    if not response.data:
        raise ValueError(f"Tweet ID {tweet_id} not found or inaccessible.")
    
    return response.data.data

@st.cache_data(show_spinner=False, ttl=600)
def fetch_tweets_by_hashtag(hashtag: str, max_results: int = 10) -> list[dict]:
    """
    Search for recent tweets containing a specific hashtag.
    """
    query = f"#{hashtag.lstrip('#')} -is:retweet lang:en"
    client = get_client()
    
    # Twitter API v2 recent search
    response = client.search_recent_tweets(
        query=query,
        max_results=max_results,
        tweet_fields=["id", "text", "created_at", "public_metrics", "author_id"]
    )
    
    if not response.data:
        return []
    
    return [tweet.data for tweet in response.data]
