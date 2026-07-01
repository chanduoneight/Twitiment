# Twitiment

Twitiment is a local sentiment analysis dashboard for text and X/Twitter content. It combines a Streamlit UI, a FastAPI API, and a deterministic VADER-based NLP pipeline.

## Features

- Streamlit dashboard for sentence, word, and aspect sentiment
- FastAPI endpoints for text, tweet URL, and hashtag analysis
- X/Twitter fetch helpers through Tweepy
- PDF export from the dashboard
- Model verification script with representative text cases

## Setup

Create a virtual environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The NLP module downloads required NLTK resources on first run.

## Credentials

Text-only analysis works without X/Twitter credentials. Tweet and hashtag fetches need API credentials in environment variables:

```bash
export BEARER_TOKEN="..."
export CONSUMER_KEY="..."
export CONSUMER_SECRET="..."
export ACCESS_TOKEN="..."
export ACCESS_TOKEN_SECRET="..."
```

You can also place these values in a local `.env` file. The repository already ignores `.env` and `.env.local`, so do not commit credentials.

## Run The Dashboard

```bash
streamlit run app.py
```

## Run The API

```bash
uvicorn api:app --reload
```

Available routes:

- `POST /api/analyze` with JSON body `{ "text": "..." }`
- `GET /api/tweet?url=<tweet-url-or-id>`
- `GET /api/hashtag?hashtag=<tag>&max_results=10`
- `GET /health`

## Verify The Pipeline

```bash
python verify_models.py
```

This runs the tokenizer, cleaner, sentence sentiment, word sentiment, and aspect sentiment paths against sample inputs.
