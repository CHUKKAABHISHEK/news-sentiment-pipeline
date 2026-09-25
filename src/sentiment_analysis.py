import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Load dataset
df = pd.read_csv("data/news.csv")


# Create sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


# Text cleaning function
def clean_text(text):
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Clean the headlines
df["cleaned_headline"] = df["headline"].apply(clean_text)


# Sentiment analysis function
def analyze_sentiment(text):

    scores = analyzer.polarity_scores(text)

    compound = scores["compound"]

    if compound >= 0.05:
        return "Positive"

    elif compound <= -0.05:
        return "Negative"

    else:
        return "Neutral"


# Analyze sentiment
df["sentiment"] = df["cleaned_headline"].apply(analyze_sentiment)


# Save processed dataset
df.to_csv(
    "data/news_with_sentiment.csv",
    index=False
)


print("Data preprocessing completed!")
print("Sentiment analysis completed!")

print("\nProcessed Data:")
print(df)

print("\nResults saved to:")
print("data/news_with_sentiment.csv")
