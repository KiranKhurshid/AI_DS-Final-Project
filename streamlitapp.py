import streamlit as st
import joblib
import numpy as np
import re
import tensorflow as tf
import keras
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

# Define the preprocess_text function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'\W', ' ', text)
    # Remove single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    return text

# Load the saved models
nb_model = joblib.load('nb_sentiment_model_word2vec.pkl')  # Load the Word2Vec Naive Bayes model
lstm_model = tf.keras.models.load_model('lstm_sentiment_model.h5')
word2vec_model = Word2Vec.load('word2vec_model.pkl')  # Load your trained Word2Vec model
tokenizer = joblib.load('tokenizer.pkl')  # Load tokenizer for LSTM model

# Streamlit app
st.title('Sentiment Analysis')

user_input = st.text_area('Enter a review:')
if st.button('Analyze'):
    # Preprocess the input
    processed_input = preprocess_text(user_input)

    # Naive Bayes prediction with Word2Vec
    def vectorize_review(review):
        words = review.split()
        word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
        if len(word_vectors) == 0:
            return np.zeros(word2vec_model.vector_size)
        return np.mean(word_vectors, axis=0)

    nb_input_vector = vectorize_review(processed_input).reshape(1, -1)  # Reshape for the model
    nb_prediction = nb_model.predict(nb_input_vector)

    # LSTM prediction
    lstm_input_seq = tokenizer.texts_to_sequences([processed_input])
    lstm_input_pad = pad_sequences(lstm_input_seq, maxlen=200)
    lstm_prediction = lstm_model.predict(lstm_input_pad)[0][0]

    # Display results
    st.write('Naive Bayes Prediction: Positive' if nb_prediction == 1 else 'Naive Bayes Prediction: Negative')
    st.write('LSTM Prediction: Positive' if lstm_prediction >= 1 else 'LSTM Prediction: Negative')
