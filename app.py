# app.py

import streamlit as st
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="IMDb Review Analyzer",
    page_icon="🎬",
    layout="wide"
)

# --- Asset Loading ---

# Function to load models and data, cached for performance
@st.cache_resource
def load_models_and_data():
    """Loads all necessary models and data from disk."""
    # Load sentiment model and vectorizer
    with open('models/sentiment_model.pkl', 'rb') as f:
        sentiment_model = pickle.load(f)
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    
    # Load LDA model and vectorizer
    with open('models/lda_model.pkl', 'rb') as f:
        lda_model = pickle.load(f)
    with open('models/count_vectorizer.pkl', 'rb') as f:
        count_vectorizer = pickle.load(f)
        
    return sentiment_model, tfidf_vectorizer, lda_model, count_vectorizer

# --- Text Preprocessing Function ---
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    stop_words = set(stopwords.words('english'))
    negation_words = {'not', 'no', 'nor', "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't"}
    stop_words = stop_words - negation_words
    
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(clean_tokens)

# --- Aspect-Based Sentiment Analysis (ABSA) Function ---
ASPECT_KEYWORDS = {
    "acting": ["actor", "actress", "acting", "performance", "cast", "character"],
    "plot": ["plot", "story", "script", "narrative", "storyline", "ending"],
    "visuals": ["visuals", "effects", "cgi", "cinematography", "scenery"],
    "directing": ["directing", "director", "filmmaker", "style"]
}

def get_aspect_sentiments(review, model, vectorizer):
    clean_review = preprocess_text(review)
    review_tokens = clean_review.split()
    aspect_sentiments = {}
    
    for aspect, keywords in ASPECT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in review_tokens:
                try:
                    keyword_index = review_tokens.index(keyword)
                    start = max(0, keyword_index - 10)
                    end = min(len(review_tokens), keyword_index + 11)
                    context_window = " ".join(review_tokens[start:end])
                    
                    vectorized_window = vectorizer.transform([context_window])
                    prediction = model.predict(vectorized_window)[0]
                    
                    aspect_sentiments[aspect] = prediction
                    break 
                except Exception as e:
                    pass
    return aspect_sentiments

# --- Main App Logic ---

# Load everything
sentiment_model, tfidf_vectorizer, lda_model, count_vectorizer = load_models_and_data()

st.title("🎬 IMDb Review Analysis Dashboard")
st.markdown("A comprehensive tool to analyze sentiment, aspects, and topics from movie reviews.")

# --- Sidebar Navigation ---
st.sidebar.title("Analysis Tools")
app_mode = st.sidebar.selectbox(
    "Choose the analysis you want to perform:",
    ["Sentiment Predictor", "Deeper Analysis (ABSA)", "Topic Explorer"]
)

# --- Page 1: Sentiment Predictor ---
if app_mode == "Sentiment Predictor":
    st.header("Overall Sentiment Prediction")
    st.markdown("Enter a movie review below to predict its overall sentiment (Positive or Negative).")
    
    user_input = st.text_area("Enter review text:", "This movie was absolutely fantastic, a must-see for everyone!", height=150)
    
    if st.button("Predict Sentiment"):
        if user_input:
            clean_input = preprocess_text(user_input)
            vectorized_input = tfidf_vectorizer.transform([clean_input])
            prediction = sentiment_model.predict(vectorized_input)[0]
            
            if prediction == "positive":
                st.success(f"Predicted Sentiment: **Positive** 👍")
            else:
                st.error(f"Predicted Sentiment: **Negative** 👎")
        else:
            st.warning("Please enter a review to analyze.")

# --- Page 2: Deeper Analysis (ABSA) ---
elif app_mode == "Deeper Analysis (ABSA)":
    st.header("Aspect-Based Sentiment Analysis")
    st.markdown("This tool breaks down a review to find the sentiment for specific aspects like **acting**, **plot**, and **visuals**.")
    
    user_input = st.text_area("Enter review text:", "The performance by the lead actor was brilliant, but the plot was predictable.", height=150)
    
    if st.button("Analyze Aspects"):
        if user_input:
            aspects = get_aspect_sentiments(user_input, sentiment_model, tfidf_vectorizer)
            
            if not aspects:
                st.info("No specific aspects (acting, plot, etc.) were mentioned in this review.")
            else:
                st.write("### Aspect Sentiment Results:")
                for aspect, sentiment in aspects.items():
                    if sentiment == "positive":
                        st.markdown(f"**{aspect.title()}:** <span style='color:green;'>**Positive**</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**{aspect.title()}:** <span style='color:red;'>**Negative**</span>", unsafe_allow_html=True)
        else:
            st.warning("Please enter a review to analyze.")

# --- Page 3: Topic Explorer ---
elif app_mode == "Topic Explorer":
    st.header("Discover Hidden Topics in Reviews")
    st.markdown("These topics were automatically discovered from the 50,000 IMDb reviews using the LDA algorithm. Each topic is represented by its most important keywords.")

    feature_names = count_vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda_model.components_):
        st.subheader(f"Topic #{topic_idx + 1}")
        topic_words = " | ".join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]])
        st.markdown(f"**Keywords:** `{topic_words}`")