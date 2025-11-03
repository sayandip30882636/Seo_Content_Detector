import pandas as pd
import regex as re
import textstat
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import streamlit as st # Import Streamlit

# --- Helper Functions ---

def safe_sentence_count(text):
    try:
        return textstat.sentence_count(text)
    except Exception:
        return 0

def safe_flesch_reading_ease(text):
    try:
        return textstat.flesch_reading_ease(text)
    except Exception:
        return 0  # 0 is a bad score, appropriate for un-parsable text

def load_tfidf_vectorizer(model_path='Seo_Content_Detector/models/tfidf_vectorizer.pkl'):
    """
    Loads a pre-trained TF-IDF vectorizer.
    """
    if not os.path.exists(model_path):
        # print(f"Warning: TF-IDF vectorizer not found at {model_path}. Returning None.")
        return None, None
    try:
        with open(model_path, 'rb') as f:
            vectorizer = pickle.load(f)
        feature_names = vectorizer.get_feature_names_out()
        return vectorizer, feature_names
    except Exception as e:
        # print(f"Error loading TF-IDF vectorizer from {model_path}: {e}")
        return None, None

def get_top_keywords(doc_vector, feature_names, top_n=5):
    """
    Extracts top N keywords from a TF-IDF document vector.
    """
    if doc_vector is None or not feature_names:
        return ""
    
    dense_vec = doc_vector.toarray().flatten()
    top_indices = dense_vec.argsort()[-top_n:][::-1]
    top_words = [feature_names[i] for i in top_indices if dense_vec[i] > 0]
    return "|".join(top_words)

@st.cache_resource
def load_sbert_model(model_name='all-MiniLM-L6-v2'):
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading SentenceTransformer model: {e}")
        return None

# --- Main Feature Extraction Function ---

def extract_all_features(body_text,
                         tfidf_vectorizer=None,
                         feature_names=None,
                         tfidf_max_features=100):
    """
    Extracts all necessary features from a given body of text.
    Includes word count, readability, TF-IDF keywords, and TF-IDF embedding.
    """
    
    # 1. Clean Text
    if pd.isna(body_text) or not body_text.strip():
        body_text_cleaned = ""
        word_count = 0
    else:
        body_text_cleaned = re.sub(r'\s+', ' ', body_text.lower().strip())
        word_count = len(body_text_cleaned.split())

    # Create a 'safe' version for textstat
    safe_text_for_stat = body_text_cleaned if word_count > 10 else "No content to analyze"

    # 2. Basic metrics
    sentence_count = safe_sentence_count(safe_text_for_stat)
    flesch_reading_ease = safe_flesch_reading_ease(safe_text_for_stat)
    is_thin = bool(word_count < 500)

    # 3. TF-IDF Keywords & Embedding
    top_keywords = ""
    embedding = [0.0] * tfidf_max_features

    if tfidf_vectorizer is None or feature_names is None:
        tfidf_vectorizer, feature_names = load_tfidf_vectorizer()

    if tfidf_vectorizer and feature_names and body_text_cleaned:
        try:
            # Ensure the vocabulary matches TFIDF_MAX_FEATURES by padding if necessary
            # This is a simplification; in a real app, ensure consistent TF-IDF training.
            doc_tfidf_matrix = tfidf_vectorizer.transform([body_text_cleaned])
            top_keywords = get_top_keywords(doc_tfidf_matrix, feature_names, top_n=5)
            
            dense_embedding = doc_tfidf_matrix.toarray()[0]
            # Pad or truncate the embedding to match tfidf_max_features
            if len(dense_embedding) < tfidf_max_features:
                embedding = np.pad(dense_embedding, (0, tfidf_max_features - len(dense_embedding)), 'constant').tolist()
            else:
                embedding = dense_embedding[:tfidf_max_features].tolist()

        except ValueError as e:
            # print(f"TF-IDF transformation error: {e}. Returning empty keywords and zero embedding.")
            top_keywords = ""
            embedding = [0.0] * tfidf_max_features
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'flesch_reading_ease': flesch_reading_ease,
        'is_thin': int(is_thin),
        'top_keywords': top_keywords,
        'embedding': embedding
    }
