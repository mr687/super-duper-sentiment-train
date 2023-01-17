import streamlit as st
import pickle as pkl
import pandas as pd
import numpy as np
import sklearn as sk

from shared import *

def load_model():
	with open('dataset/model/sental_model.pkl', 'rb') as f:
		model = pkl.load(f)
		return model

def load_vectorizer():
	with open('dataset/model/sental_vectorizer.pkl', 'rb') as f:
		vectorizer = pkl.load(f)
		return vectorizer

def get_prediction(tweet_text):
	model = load_model()
	vectorizer = load_vectorizer()
	tweet_vector = vectorizer.transform([tweet_text])
	subjectivity = model.predict(tweet_vector)
	if subjectivity == 0:
		return 'Negative'
	else:
		return 'Positive'

st.title("Sentiment Analysis of Tweets about Indonesia's Fuel Price Hike")

st.header("Enter a Tweet below")

tweet_text = st.text_area('Tweet Text')

if st.button('Analyze'):
	st.header("Sentiment Analysis Results")

	text = preprocess_text(tweet_text)

	result = get_prediction(text)

	st.write(f"Tweet: {tweet_text}")
	st.write(f"Preprocessed Tweet: {text}")

	if result == 'Negative':
		st.error(f"Sentiment: {result}")
	else:
		st.success(f"Sentiment: {result}")
