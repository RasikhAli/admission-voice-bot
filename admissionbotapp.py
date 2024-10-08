from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re
from langdetect import detect
from deep_translator import GoogleTranslator

app = Flask(__name__)

# Load the admissions queries dataset and embeddings
admission_df = pd.read_csv('data/admission_queries_responses3.csv')
admission_embeddings = np.load('data/admission_query_embeddings.npy')

# Load the SentenceTransformer model for admission queries
model_path_admission = 'LLM/paraphrase-MiniLM-L6-v2'
model_admission = SentenceTransformer(model_path_admission)

# Load FAISS index for admission queries
index_admission = faiss.read_index('data/admission_faiss.index')

# Function to clean text
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

# Function to detect the language
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'  # Default to English if detection fails

# Function to retrieve similar admissions queries
def retrieve_similar_queries(query, model, index, df, k=5):
    query_embedding = model.encode([clean_text(query)])
    distances, indices = index.search(query_embedding, k)
    
    results = []
    for i in range(k):
        index_pos = indices[0][i]
        context_text = df['Context'].iloc[index_pos] if not pd.isna(df['Context'].iloc[index_pos]) else 'N/A'
        query_text = df['Query'].iloc[index_pos] if not pd.isna(df['Query'].iloc[index_pos]) else 'N/A'
        response_text = df['Response'].iloc[index_pos] if not pd.isna(df['Response'].iloc[index_pos]) else 'N/A'
        
        query_info = {
            'query': query_text,
            'response': response_text,
            'distance': float(distances[0][i]),
            'context': context_text
        }
        results.append(query_info)

    if not results:
        return [{'response': "Couldn't find matching queries."}]

    return results

# Route for the homepage
@app.route('/')
def home():
    return render_template('admissionindex.html')

# Route for handling queries related to admissions
@app.route('/get_similar_queries', methods=['POST'])
def get_similar_queries():
    user_input = request.form['query'].strip().lower()  # Normalize input

    if user_input == 'stop' or user_input == 'ok stop':
        return jsonify([])  # Return an empty list to indicate stopping

    lang = detect_language(user_input)

    if lang == 'ar':
        translated_query = GoogleTranslator(source='ar', target='en').translate(user_input)
        results = retrieve_similar_queries(translated_query, model_admission, index_admission, admission_df, k=1)
    elif lang == 'ur':
        translated_query = GoogleTranslator(source='ur', target='en').translate(user_input)
        results = retrieve_similar_queries(translated_query, model_admission, index_admission, admission_df, k=1)
        for result in results:
            result['response'] = GoogleTranslator(source='en', target='ur').translate(result['response'])
    else:
        results = retrieve_similar_queries(user_input, model_admission, index_admission, admission_df, k=1)

    return jsonify(results)  # Return the results as JSON

if __name__ == '__main__':
    app.run(debug=True)
