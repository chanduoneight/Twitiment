import sys
import os

# Ensure terminal output handles UTF-8 (emojis etc)
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from sentiment_engine import (
    get_clean_text,
    sentence_sentiment,
    word_sentiment,
    aspect_sentiment,
)

def test_models():
    test_cases = [
        "I absolutely love this new feature! It's amazing. 😍",
        "I hate the new update, it's terrible and slow. 😡",
        "The weather is okay today, nothing special.",
        "Check out this link: http://example.com/test",
        "<div>HTML tags should be stripped.</div>",
        "Python is great for data science, but some libraries are hard to learn.",
    ]

    print("=== Sentiment Model Verification ===\n")

    for i, text in enumerate(test_cases, 1):
        print(f"Test Case {i}: {text}")
        
        # 1. Cleaning
        clean = get_clean_text(text)
        print(f"  [Clean Text]: {clean}")

        # 2. Sentence-Level (VADER)
        sent = sentence_sentiment(clean)
        print(f"  [Sentence-Level]: {sent['label']} (Compound: {sent['compound']:.4f})")

        # 3. Word-Level
        words = word_sentiment(clean)
        print(f"  [Word-Level Hits]: {[(w['word'], w['compound']) for w in words if w['label'] != 'neutral']}")

        # 4. Aspect-Based
        aspects = aspect_sentiment(clean)
        print(f"  [Aspects]: {[(a['aspect'], a['label'], a['compound']) for a in aspects]}")
        print("-" * 40)

if __name__ == "__main__":
    try:
        test_models()
        print("\nVerification completed successfully.")
    except Exception as e:
        print(f"\nVerification failed with error: {e}")
        sys.exit(1)
