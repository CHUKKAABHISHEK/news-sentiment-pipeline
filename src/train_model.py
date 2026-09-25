import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# Load dataset
df = pd.read_csv("data/news_with_sentiment.csv")


# Input and output
X = df["cleaned_headline"]
y = df["sentiment"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Create ML model
model = LogisticRegression(
    max_iter=1000
)


# Train model
model.fit(
    X_train_tfidf,
    y_train
)


# Predictions
y_pred = model.predict(
    X_test_tfidf
)


# Accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("===================================")
print("   NEWS SENTIMENT ML MODEL")
print("===================================")

print()
print("Accuracy:", f"{accuracy * 100:.2f}%")

print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# Confusion Matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Negative", "Neutral", "Positive"]
)

print()
print("Confusion Matrix:")
print(cm)


# Display Confusion Matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Neutral", "Positive"]
)

display.plot()

plt.title("News Sentiment Confusion Matrix")

plt.savefig(
    "models/confusion_matrix.png"
)

plt.show()


# Save model
joblib.dump(
    model,
    "models/sentiment_model.pkl"
)


# Save vectorizer
joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)


print()
print("===================================")
print("Model saved successfully!")
print("Vectorizer saved successfully!")
print("Confusion matrix saved successfully!")
print("===================================")
