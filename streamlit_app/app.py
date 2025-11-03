import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import torch
import json
import warnings

# Import utility functions from local files
from Seo_Content_Detector.streamlit_app.utils.parser import parse_html_regex
from Seo_Content_Detector.streamlit_app.utils.features import extract_all_features, load_sbert_model, load_tfidf_vectorizer
from Seo_Content_Detector.streamlit_app.utils.scorer import analyze_url, train_quality_model, apply_quality_label, safe_eval

# Suppress warnings for cleaner output in Streamlit
warnings.filterwarnings("ignore")

# --- Configuration Constants ---
PROJECT_ROOT = 'Seo_Content_Detector'
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'Data')

# File paths for models and data
QUALITY_MODEL_PATH = os.path.join(MODELS_DIR, 'quality_model.pkl')
TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
EMBEDDINGS_NPY_PATH = os.path.join(MODELS_DIR, 'sentence_embeddings.npy')

EXTRACTED_CONTENT_CSV = os.path.join(DATA_DIR, 'extracted_content.csv')
FEATURES_CSV = os.path.join(DATA_DIR, 'features.csv')
DUPLICATES_CSV = os.path.join(DATA_DIR, 'duplicate.csv')

TFIDF_MAX_FEATURES = 100 # Must match the value used during feature extraction

# --- Cached Resources ---
@st.cache_resource
def load_quality_model():
    if os.path.exists(QUALITY_MODEL_PATH):
        with open(QUALITY_MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_resource
def load_sbert():
    return load_sbert_model()

@st.cache_resource
def load_tfidf_assets():
    return load_tfidf_vectorizer(model_path=TFIDF_VECTORIZER_PATH)

@st.cache_resource
def load_corpus_embeddings_and_urls():
    if os.path.exists(EMBEDDINGS_NPY_PATH) and os.path.exists(FEATURES_CSV):
        corpus_embeddings = np.load(EMBEDDINGS_NPY_PATH)
        df_features = pd.read_csv(FEATURES_CSV)
        known_urls = df_features['url'].tolist()
        return torch.tensor(corpus_embeddings), known_urls
    return None, []

@st.cache_data
def load_dataframes():
    df_extracted = None
    df_features = None
    df_duplicates = None

    if os.path.exists(EXTRACTED_CONTENT_CSV):
        df_extracted = pd.read_csv(EXTRACTED_CONTENT_CSV)

    if os.path.exists(FEATURES_CSV):
        df_features = pd.read_csv(FEATURES_CSV)
        # Ensure 'is_thin' exists for quality labeling
        if 'is_thin' not in df_features.columns:
            df_features['is_thin'] = (df_features['word_count'] < 500).astype(int)
        df_features['quality_label'] = df_features.apply(apply_quality_label, axis=1)

    if os.path.exists(DUPLICATES_CSV):
        df_duplicates = pd.read_csv(DUPLICATES_CSV)

    return df_extracted, df_features, df_duplicates


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="SEO Content Detector")

st.title("SEO Content Quality Detector")

# Load all assets and dataframes
quality_model = load_quality_model()
sbert_model = load_sbert()
tfidf_vectorizer, tfidf_feature_names = load_tfidf_assets()
corpus_embeddings, known_urls = load_corpus_embeddings_and_urls()
df_extracted, df_features, df_duplicates = load_dataframes()

# Check if models/embeddings are trained
models_trained = quality_model is not None and sbert_model is not None and                  tfidf_vectorizer is not None and corpus_embeddings is not None and                  len(known_urls) > 0

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "Data Overview & Parsing",
    "Feature Extraction",
    "Duplicate Detection",
    "Quality Model Overview",
    "Real-Time Analysis Demo"
])

if page == "Home":
    st.header("Welcome to the SEO Content Quality Detector")
    st.markdown('''
This Streamlit application helps analyze web content for SEO quality by extracting features,
dectecting duplicates, and predicting content quality using a trained machine learning model.

**To get started, please ensure all necessary models and data are generated/trained.**
''')

    if not models_trained:
        st.warning("It seems the quality model or corpus embeddings are not trained/loaded. Please train them before proceeding to analysis sections.")
        if st.button("Train/Retrain Quality Model & Embeddings"):
            with st.spinner("Training model and generating corpus embeddings... This might take a moment."):
                try:
                    train_quality_model()
                    # Clear caches to reload newly trained models
                    load_quality_model.clear()
                    load_sbert.clear()
                    load_tfidf_assets.clear()
                    load_corpus_embeddings_and_urls.clear()
                    st.success("Models trained and assets refreshed! You can now navigate to other sections.")
                    st.experimental_rerun() # Rerun to ensure new models are loaded
                except Exception as e:
                    st.error(f"Error during training: {e}")
    else:
        st.success("All required models and data assets are loaded. You can now perform content analysis.")

elif page == "Data Overview & Parsing":
    st.header("1. Data Overview & Parsing")
    st.write("This section displays the raw and parsed HTML content from the initial data extraction.")

    if df_extracted is not None:
        st.subheader("Extracted Content (extracted_content.csv)")
        st.dataframe(df_extracted)
    else:
        st.warning(f"'{EXTRACTED_CONTENT_CSV}' not found. Please run the data parsing step.")

elif page == "Feature Extraction":
    st.header("2. Feature Extraction")
    st.write("Here you can see the engineered features, including word counts, readability scores, and TF-IDF information.")

    if df_features is not None:
        st.subheader("Engineered Features (features.csv)")
        st.dataframe(df_features)
    else:
        st.warning(f"'{FEATURES_CSV}' not found. Please run the feature extraction step.")

elif page == "Duplicate Detection":
    st.header("3. Duplicate Detection")
    st.write("This section highlights potential duplicate or highly similar content pages based on TF-IDF cosine similarity.")

    if df_duplicates is not None:
        st.subheader("Duplicate Content Pairs (duplicate.csv)")
        st.dataframe(df_duplicates)

        total_pages = len(df_features) if df_features is not None else 0
        duplicate_pairs_count = len(df_duplicates)
        thin_content_count = df_features['is_thin'].sum() if df_features is not None else 0
        thin_content_percent = (thin_content_count / total_pages) * 100 if total_pages > 0 else 0

        st.markdown("### Summary of Content Issues")
        st.write(f"Total pages analyzed: {total_pages}")
        st.write(f"Duplicate pairs found (Semantic Similarity > 0.8): {duplicate_pairs_count}")
        st.write(f"Thin content pages (Word Count < 500): {thin_content_count} ({thin_content_percent:.1f}%) ")

    else:
        st.warning(f"'{DUPLICATES_CSV}' not found. Please run the duplicate detection step.")

elif page == "Quality Model Overview":
    st.header("4. Content Quality Model Overview")
    st.write("This section displays the features used for quality prediction and the assigned quality labels.")

    if models_trained and df_features is not None:
        st.subheader("Content Features with Quality Labels")
        st.dataframe(df_features[['url', 'word_count', 'sentence_count', 'flesch_reading_ease', 'is_thin', 'quality_label']])

        st.subheader("Quality Label Distribution")
        st.write(df_features['quality_label'].value_counts().to_frame())

    elif not models_trained:
        st.warning("Models not trained. Please go to 'Home' and train the quality model.")
    else:
        st.warning(f"'{FEATURES_CSV}' not found. Please run the feature extraction step.")

elif page == "Real-Time Analysis Demo":
    st.header("5. Real-Time URL Analysis")
    st.write("Enter a URL below to get an on-the-fly analysis of its content quality and semantic similarity to existing data.")

    input_url = st.text_input("Enter a URL", "https://www.kaspersky.com/resource-center/definitions/what-is-cyber-security")

    if st.button("Analyze URL"):
        if input_url:
            if models_trained:
                with st.spinner("Analyzing URL..."):
                    # Pass all loaded assets to the analyze_url function
                    analysis_result = analyze_url(
                        input_url,
                        sbert_model,
                        quality_model,
                        known_urls,
                        corpus_embeddings,
                        tfidf_vectorizer,
                        tfidf_feature_names
                    )

                    if "error" in analysis_result:
                        st.error(f"Analysis failed: {analysis_result['error']}")
                    else:
                        st.subheader("Analysis Results (JSON)")
                        st.json(analysis_result)

                        st.subheader("Interpretation")
                        st.write(f"**URL:** {analysis_result['url']}")
                        st.write(f"**Title:** {analysis_result['title']}")
                        st.write(f"**Word Count:** {analysis_result['word_count']}")
                        st.write(f"**Readability (Flesch Reading Ease):** {analysis_result['readability']}")
                        st.write(f"**Predicted Quality Label:** {analysis_result['quality_label']}")
                        st.write(f"**Is Thin Content (Word Count < 500):** {'Yes' if analysis_result['is_thin'] else 'No'}")

                        if analysis_result['similar_to']:
                            st.markdown("**Top 5 Semantically Similar Pages in Corpus:**")
                            for item in analysis_result['similar_to']:
                                st.write(f"- URL: {item['url']} (Similarity: {item['similarity']:.2f})")
                        else:
                            st.info("No semantically similar pages found in the corpus above the threshold (0.60).")
            else:
                st.warning("Please ensure the quality model and embeddings are trained/loaded before analyzing URLs. Go to 'Home' to train them.")
        else:
            st.warning("Please enter a URL to analyze.")
