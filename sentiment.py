import os
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"

from transformers import pipeline

# Correct way to initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline(
    task="sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def get_sentiment(text):
    """
    Get the sentiment of the input text:
    Returns:
        sentiment: 'POSITIVE' or 'NEGATIVE'
        score: confidence score of the sentiment
    """
    result = sentiment_pipeline(text)
    sentiment = result[0]['label']
    score = result[0]['score']
    return sentiment, score

# Optional test
if __name__ == "__main__":
    sentiment, score = get_sentiment("i am very happy today")
    print(f"Sentiment: {sentiment}, Score: {score:.2f}")
