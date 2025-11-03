# SEO Content Quality & Duplicate Detector

This project is an end-to-end data science pipeline that analyzes web content to assess its SEO quality and detect duplicate articles. The entire analysis, from parsing raw HTML to training a machine learning model, is captured in the `notebooks/seo_pipeline.ipynb`.

The project also includes a multi-page Streamlit web application that provides a full UI for the pipeline, including a real-time analysis tool for any live URL.



---

## 🚀 Key Features

This project is built as a 5-step pipeline:

* **1. HTML Parsing:** Reads a raw dataset (`data.csv`) containing HTML content and parses it using regex to extract clean body text, titles, and word counts.
* **2. Feature Engineering:** Enriches the clean text by calculating key metrics like Flesch Reading Ease (readability), sentence count, and top keywords (using TF-IDF).
* **3. Duplicate & Thin Content Detection:**
    * Flags "thin content" pages (word count < 500) which are often poor for SEO.
    * Uses TF-IDF embeddings and cosine similarity to find and report near-duplicate articles within the dataset.
* **4. Quality Classification:**
    * Creates synthetic labels (Low, Medium, High) based on content heuristics (e.g., word count, readability).
    * Trains a Random Forest classifier to predict these quality labels.
    * **Result:** The model achieved **96% accuracy**, far surpassing a simple rule-based (word count only) baseline model which scored **64%**.
* **5. Real-Time Analysis Demo:**
    * A live analysis tool that can scrape **any external URL**.
    * Uses **Sentence Transformers** (`all-MiniLM-L6-v2`) for powerful *semantic* search, allowing it to find conceptually similar articles in our dataset, even if they don't share the same keywords.

---

## 🛠️ Tech Stack

* **Data Analysis & ML:** Python, Pandas, Numpy, Scikit-learn
* **NLP & Scraping:** Sentence-Transformers, Textstat, BeautifulSoup4, Requests, Regex
* **Web Application:** Streamlit
* **Original Analysis:** Jupyter Notebook

---

## 📂 Project Structure

```text
seo-content-detector/
├── data/
│   ├── data.csv              # Provided raw dataset
│   ├── extracted_content.csv # Step 1 output
│   ├── features.csv          # Step 2 & 3 output
│   └── duplicate.csv         # Step 3 output
├── notebooks/
│   └── seo_pipeline.ipynb    # Original notebook analysis
├── streamlit_app/
│   ├── app.py                # Main Streamlit app
│   ├── utils/
│   │   ├── parser.py         # Step 1 logic
│   │   ├── features.py     # Step 2 & 3 logic
│   │   └── scorer.py         # Step 4 & 5 logic
│   └── models/
│       ├── quality_model.pkl   # Step 4 output
│  
└── requirements.txt
└── sentence_embeddings.py
