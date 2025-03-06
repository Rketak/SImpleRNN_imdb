import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load pre-trained model
model = load_model('simple_rnn_imdb.keras')

# Function to preprocess text
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit UI
st.set_page_config(page_title="IMDB Sentiment Analysis", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        .main { background-color: #f5f7fa; }
        .stTextArea textarea { background-color: #fff; font-size: 18px; padding: 10px; }
        .stButton button { background-color: #ff4b4b; color: white; font-size: 18px; padding: 10px; border-radius: 10px; }
        .sentiment-box { text-align: center; font-size: 22px; font-weight: bold; margin-top: 20px; padding: 10px; border-radius: 10px; }
        .positive { background-color: #d4edda; color: #155724; }
        .negative { background-color: #f8d7da; color: #721c24; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center;'>🎬 IMDB Movie Review Sentiment Analysis</h1>", unsafe_allow_html=True)
st.write("🔍 **Enter a movie review below and find out whether it's Positive or Negative!**")

# User input
user_input = st.text_area("📝 Type your movie review here:", height=150)

# Button for classification
if st.button("🔮 Classify Sentiment"):
    if user_input.strip():  # Check if input is not empty
        preprocessed_input = preprocess_text(user_input)

        # Make prediction
        prediction = model.predict(preprocessed_input)
        sentiment = "Positive 😃" if prediction[0][0] > 0.5 else "Negative 😞"
        sentiment_class = "positive" if prediction[0][0] > 0.5 else "negative"

        # Display the result
        st.markdown(f'<div class="sentiment-box {sentiment_class}">Sentiment: {sentiment}<br>Confidence Score: {prediction[0][0]:.4f}</div>',
                    unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a movie review before classifying.")

