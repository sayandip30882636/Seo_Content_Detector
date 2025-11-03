import pandas as pd
import numpy as np
import pickle
import os
import requests
from bs4 import BeautifulSoup
import warnings
import ast
import json
import textstat

# Scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Import utility functions from local files
from .parser import parse_html_regex
from .features import extract_all_features, load_sbert_model, load_tfidf_vectorizer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Configuration Constants ---
PROJECT_ROOT = 'Seo_Content_Detector'
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'Data')

MODEL_FILE = os.path.join(MODELS_DIR, 'quality_model.pkl')
TEXT_FILE = os.path.join(DATA_DIR, 'extracted_content.csv')
FEATURES_FILE = os.path.join(DATA_DIR, 'features.csv')
EMBEDDING_FILE = os.path.join(MODELS_DIR, 'sentence_embeddings.npy') # Changed to models directory
TFIDF_MAX_FEATURES = 100

# --- Helper Functions ---

def apply_quality_label(row):
    """
    Applies a quality label (High, Medium, Low) based on word count and readability.
    """
    word_count = row['word_count']
    readability = row['flesch_reading_ease']

    # Rule 1: High Quality
    if word_count > 1500 and (readability >= 50 and readability <= 70):
        return "High"
    # Rule 2: Low Quality
    # (Uses 'is_thin' for word_count < 500 or very low readability)
    elif row['is_thin'] == 1 or readability < 30:
        return "Low"
    # Rule 3: Medium Quality (all other cases)
    else:
        return "Medium"

def safe_eval(s):
    """
    Safely evaluates a string as a Python literal (e.g., a list or dict).
    Handles empty or malformed strings by returning an empty list.
    """
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError):
        return []

# --- Main Functions ---

def train_quality_model():
    """
    Trains a RandomForestClassifier for content quality and saves the model
    along with corpus embeddings and known URLs.
    """
    print("Starting training of quality model...")

    if not os.path.exists(FEATURES_FILE):
        print(f"Error: Features file '{FEATURES_FILE}' not found. Cannot train model.")
        return

    df = pd.read_csv(FEATURES_FILE)
    df_text = pd.read_csv(TEXT_FILE) # To get body_text for SBERT embedding

    # Ensure 'is_thin' column exists (it should from previous step)
    if 'is_thin' not in df.columns:
        df['is_thin'] = (df['word_count'] < 500).astype(int)

    df['quality_label'] = df.apply(apply_quality_label, axis=1)

    feature_cols = ['word_count', 'sentence_count', 'flesch_reading_ease', 'is_thin']
    X = df[feature_cols]
    y = df['quality_label']

    # Handle cases where a label might have only one sample
    unique_labels, counts = np.unique(y, return_counts=True)
    stratify_y = y if min(counts) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=stratify_y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    print(f"Quality model saved to '{MODEL_FILE}'")

    # Pre-compute and save SBERT embeddings for similarity search
    sbert_model = load_sbert_model() # Using cached model from features.py
    if sbert_model:
        corpus_texts = df_text['body_text'].fillna('').astype(str).tolist()
        corpus_embeddings = sbert_model.encode(corpus_texts, convert_to_tensor=True)
        np.save(EMBEDDING_FILE, corpus_embeddings.cpu().numpy())
        print(f"Corpus embeddings saved to '{EMBEDDING_FILE}'")

    print("Training and asset saving complete.")


def analyze_url(url, sbert_model, quality_model, known_urls, corpus_embeddings, tfidf_vectorizer, tfidf_feature_names):
    """
    Scrapes a live URL, extracts features, predicts quality,
    and finds SEMANTICALLY similar matches from our existing dataset.
    """

    # --- Part A: Scrape the URL ---
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return {"error": f"Failed to fetch URL. Status code: {response.status_code}"}

        html_content = response.text
    except Exception as e:
        return {"error": f"Scraping failed: {str(e)}"}

    # --- Part B: Parse and Engineer Features ---
    try:
        # Use the parse_html_regex for consistency with original parsing
        title, body_text, _ = parse_html_regex(html_content)

        # Extract all features using the utility function
        features = extract_all_features(body_text, tfidf_vectorizer, tfidf_feature_names, TFIDF_MAX_FEATURES)
        word_count = features['word_count']
        sentence_count = features['sentence_count']
        readability = features['flesch_reading_ease']
        is_thin = features['is_thin']
        new_text_embedding = features['embedding']

    except Exception as e:
        return {"error": f"Feature extraction failed: {str(e)}"}

    # --- Part C: Predict Quality ---
    feature_cols = ['word_count', 'sentence_count', 'flesch_reading_ease', 'is_thin']
    features_df = pd.DataFrame(
        [[word_count, sentence_count, readability, int(is_thin)]],
        columns=feature_cols
    )
    quality_label = quality_model.predict(features_df)[0]

    # --- Part D: Detect Duplicates (Semantic Similarity) ---
    SEMANTIC_SIMILARITY_THRESHOLD = 0.60 # Can be adjusted

    similar_to = []
    if sbert_model and corpus_embeddings is not None and len(corpus_embeddings) > 0:
        try:
            new_embedding_tensor = sbert_model.encode([body_text], convert_to_tensor=True)
            # Ensure corpus_embeddings is a tensor if it was loaded as numpy array
            if isinstance(corpus_embeddings, np.ndarray):
                import torch
                corpus_embeddings = torch.tensor(corpus_embeddings)
            
            similarities = cosine_similarity(new_embedding_tensor.cpu().numpy(), corpus_embeddings.cpu().numpy())
            
            # Flatten the similarities array to easily iterate through scores
            sim_scores = similarities.flatten()

            for i, sim_score in enumerate(sim_scores):
                if sim_score > SEMANTIC_SIMILARITY_THRESHOLD and sim_score < 0.99: # Exclude self-similarity or exact matches
                    similar_to.append({
                        "url": known_urls[i],
                        "similarity": round(float(sim_score), 2)
                    })

            similar_to.sort(key=lambda x: x['similarity'], reverse=True)
        except Exception as e:
            print(f"Semantic similarity calculation failed: {e}")


    # --- Part E: Format the Final Output ---
    result = {
        "url": url,
        "title": title,
        "word_count": word_count,
        "readability": round(readability, 2),
        "quality_label": quality_label,
        "is_thin": bool(is_thin),
        "similar_to": similar_to[:5] if similar_to else [] # Show top 5 matches
    }

    return result
