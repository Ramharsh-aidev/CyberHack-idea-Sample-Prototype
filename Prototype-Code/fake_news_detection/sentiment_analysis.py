# fake_news_detection/sentiment_analysis.py
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure you have the necessary NLTK data files
nltk.download('vader_lexicon')

def analyze_sentiment(text):
    """Analyze sentiment of the given text using VADER sentiment analyzer."""
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)
    
    # Sentiment score analysis
    if sentiment_score['compound'] >= 0.05:
        sentiment = 'Positive'
    elif sentiment_score['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return sentiment, sentiment_score['compound']
