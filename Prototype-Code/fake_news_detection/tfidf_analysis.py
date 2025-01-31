# fake_news_detection/tfidf_analysis.py
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def analyze_tfidf(news_texts):
    """Analyze the given news articles using TF-IDF to identify key terms."""
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=20)
    
    # Fit and transform the news articles to obtain the TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(news_texts)

    # Create a DataFrame to display the TF-IDF results
    feature_names = vectorizer.get_feature_names_out()
    dense = tfidf_matrix.todense()
    df_tfidf = pd.DataFrame(dense, columns=feature_names)
    
    return df_tfidf
