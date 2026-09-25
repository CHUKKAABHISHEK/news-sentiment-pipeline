import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="News Sentiment Dashboard",
    page_icon="📰",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------

st.title("📰 News Sentiment Analysis Dashboard")

st.write(
    "A Data Science Pipeline for analyzing news headlines "
    "using Machine Learning."
)


# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv(
    "data/news_with_sentiment.csv"
)


# -----------------------------
# Load ML Model
# -----------------------------

model = joblib.load(
    "models/sentiment_model.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# -----------------------------
# Dashboard Statistics
# -----------------------------

total_news = len(df)

positive_news = len(
    df[df["sentiment"] == "Positive"]
)

negative_news = len(
    df[df["sentiment"] == "Negative"]
)

neutral_news = len(
    df[df["sentiment"] == "Neutral"]
)


# -----------------------------
# Dashboard Cards
# -----------------------------

st.subheader("📊 Dashboard Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total News",
    total_news
)

col2.metric(
    "🟢 Positive",
    positive_news
)

col3.metric(
    "🔴 Negative",
    negative_news
)

col4.metric(
    "🟡 Neutral",
    neutral_news
)


# -----------------------------
# Sentiment Percentages
# -----------------------------

st.subheader("📈 Sentiment Percentages")

positive_percent = (
    positive_news / total_news
) * 100

negative_percent = (
    negative_news / total_news
) * 100

neutral_percent = (
    neutral_news / total_news
) * 100


col1, col2, col3 = st.columns(3)

col1.metric(
    "Positive %",
    f"{positive_percent:.1f}%"
)

col2.metric(
    "Negative %",
    f"{negative_percent:.1f}%"
)

col3.metric(
    "Neutral %",
    f"{neutral_percent:.1f}%"
)


# -----------------------------
# Sentiment Chart
# -----------------------------

st.subheader("📊 Sentiment Distribution")

sentiment_counts = (
    df["sentiment"]
    .value_counts()
)


col1, col2 = st.columns(2)


with col1:

    st.write("### Bar Chart")

    st.bar_chart(
        sentiment_counts
    )


with col2:

    st.write("### Pie Chart")

    fig, ax = plt.subplots()

    ax.pie(
        sentiment_counts.values,
        labels=sentiment_counts.index,
        autopct="%1.1f%%"
    )

    ax.set_title(
        "Sentiment Distribution"
    )

    st.pyplot(fig)


# -----------------------------
# Dataset
# -----------------------------

st.subheader("📰 News Dataset")

st.dataframe(
    df,
    use_container_width=True
)


# -----------------------------
# ML Prediction
# -----------------------------

st.subheader(
    "🤖 Predict Sentiment Using ML Model"
)

news = st.text_input(
    "Enter a new news headline:"
)


if st.button("Predict Sentiment"):

    if news.strip() == "":

        st.warning(
            "Please enter a news headline."
        )

    else:

        # Convert text to TF-IDF
        news_tfidf = vectorizer.transform(
            [news]
        )

        # Predict sentiment
        prediction = model.predict(
            news_tfidf
        )[0]


        # Display result

        if prediction == "Positive":

            st.success(
                "🟢 Positive Sentiment"
            )

        elif prediction == "Negative":

            st.error(
                "🔴 Negative Sentiment"
            )

        else:

            st.info(
                "🟡 Neutral Sentiment"
            )


        st.write(
            "ML Model Prediction:",
            prediction
        )
        