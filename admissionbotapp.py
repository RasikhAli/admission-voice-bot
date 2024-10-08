from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re
from langdetect import detect
from deep_translator import GoogleTranslator
import pyttsx3
import speech_recognition as sr
from twilio.rest import Client  # For WhatsApp integration


app = Flask(__name__)

# WhatsApp configurations (to be filled with your details)
TWILIO_SID = 'ACe9abe5a7eb62e341cd1c5df94bb08e6f'
TWILIO_AUTH_TOKEN = 'b590d01d991e27b1a5ef99dab9264c91'
TWILIO_WHATSAPP_NUMBER = 'whatsapp:+14155238886'

# Load the admissions queries dataset and embeddings
admission_df = pd.read_csv('data/admission_queries_responses3.csv')
admission_embeddings = np.load('data/admission_query_embeddings.npy')

# Load the SentenceTransformer model for admission queries
model_path_admission = 'LLM/paraphrase-MiniLM-L6-v2'
model_admission = SentenceTransformer(model_path_admission)

# Load FAISS index for admission queries
index_admission = faiss.read_index('data/admission_faiss.index')

# Initialize the TTS engine (Text-to-Speech)
engine = pyttsx3.init()

# Set properties for voice
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume level

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



@app.route('/send_whatsapp', methods=['POST']) 
def send_whatsapp():
    data = request.get_json()  # Use get_json to get the JSON data
    whatsapp_number = data.get('whatsapp_number')
    whatsapp_number = whatsapp_number.replace("+", '').replace(" ", '')

    gmessage = ("Assalam o Alaikum Sir/Ma'am,\n\n"
                    "(I hope you're doing well)\n\n"
                    "I'm Admission Bot, and here are the links to access the relevant details:\n"
                    "- Website: https://www.sheffield.ac.uk/\n"
                    "- Admissions: https://www.sheffield.ac.uk/study/admissions\n"
                    "- Courses: https://www.sheffield.ac.uk/courses\n"
                    "- Scholarship: https://usic.sheffield.ac.uk/how-to-apply/scholarships\n"
                    "- Fee Structure: https://usic.sheffield.ac.uk/how-to-apply/fees\n\n"
                    "JazakAllah,\n"
                    "Allah Hafiz.")
    
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    try:
        message = client.messages.create(
            body=gmessage,  # Ensure the message variable is defined
            from_=TWILIO_WHATSAPP_NUMBER,
            to=f'whatsapp:+{whatsapp_number}'
        )
        print(f"WhatsApp message sent to {whatsapp_number}")
        return jsonify({"status": "success", "message": "WhatsApp details sent."}), 200
    except Exception as e:
        print(f"Failed to send WhatsApp message. Error: {e}")
        return jsonify({"status": "failed", "message": "WhatsApp details not sent."}), 500


if __name__ == '__main__':
    app.run(debug=True)
