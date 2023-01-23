import streamlit as st
import pickle as pkl
import pandas as pd
import numpy as np
import sklearn as sk
import nltk
import swifter

nltk.download('punkt')
nltk.download('stopwords')

from shared import *

def error_response(st, message):
	st.error(message)

def success_response(st, message):
	st.success(message)

def info_response(st, message):
	st.info(message)

def load_model():

	with open('dataset/model/sental_model.pkl', 'rb') as f:
		model = pkl.load(f)
		return model

def load_vectorizer():
	with open('dataset/model/sental_vectorizer.pkl', 'rb') as f:
		vectorizer = pkl.load(f)
		return vectorizer

float_to_percent = lambda x: f"{x*100:.1f}%"

model = load_model()
vectorizer = load_vectorizer()
def get_prediction(tweet_text):
	tweet_vector = vectorizer.transform([tweet_text])
	subjectivity = model.predict(tweet_vector)
	probabilities = model.predict_proba(tweet_vector)
	return subjectivity, probabilities

def determine_sentiment(subjectivity):
	return 'Negative' if subjectivity[0] == 0 else 'Positive'

def write_result(st, text_ori, text_clean, result):
	subjectivity, probabilities = result
	negative_proba = probabilities[0][0]
	positive_proba = probabilities[0][1]
	sentiment = determine_sentiment(subjectivity)
	
	info_response(st, (f"""
	Text Preprocessing
	- Original Text: `{text_ori}`
	- Preprocessed Text: `{text_clean}`
	"""))
	
	info_response(st, (f"""
	Polarity score
	- Negative score: `{negative_proba}` ({float_to_percent(negative_proba)})
	- Positive score: `{positive_proba}` ({float_to_percent(positive_proba)})
	"""))

	if sentiment == 'Negative':
		error_response(st, f"Sentiment: `{sentiment}`")
	else:
		success_response(st, f"Sentiment: `{sentiment}`")

st.title("Sentiment Analysis of Tweets about Indonesia's Fuel Price Hike")
st.write("This is a simple sentiment analysis of tweets about Indonesia's fuel price hike developed by [Davi Nomoeh Dani](https://mr687.github.io). The model is trained using a SVM classifier and Lexicon-based approach.")
st.write("---")

def handle_uploaded_file(st, file=None, delim=',', column='text'):
	if file is None:
			error_response(st, 'Please upload a file')
			return
	try:
		df = pd.read_csv(file, delimiter=delim)
	except Exception as e:
		error_response(st, 'Error reading the file. Please check the file format and delimiter.')
		return
	if column not in df.columns:
		error_response(st, f"Column '{column}' not found in the file. Available columns: `{', '.join(df.columns)}`")
		return
	with st.spinner('Preparing data...'):
		df = df[[column]]
		total = len(df)
		info_response(st, f"""
		Raw Dataset Information
		- Total Rows: `{total}`
		- Word Count: `{df[column].swifter.apply(lambda x: len(x.split())).sum()}`
		""")
	
	progress_bar = st.progress(0)
	with st.spinner('Preprocessing data...'):
		df = preprocessing(df, column=column)
		progress_bar.progress(10)
	with st.spinner('Normalizing data...'):
		df = normalizer(df, column='tokens')
		progress_bar.progress(15)
	with st.spinner('Cleaning data...'):
		threshold = int(df.shape[0] * 0.1) # threshold is 10% of the total data
		df = clean_data(df, column='tokens', threshold=threshold)
		progress_bar.progress(30)
	with st.spinner('Stemming data...'):
		df = stemming(df, column='tokens')
		progress_bar.progress(70)
	with st.spinner('Predicting data...'):
		df['subjectivity'], df['probabilities'] = zip(*df['review'].swifter.apply(get_prediction))
		progress_bar.progress(95)
		df['sentiment'] = df['subjectivity'].swifter.apply(determine_sentiment)
		progress_bar.progress(100)

	df = df[[column, 'sentiment']]
	st.write(df)

with st.expander('Manual Input'):
	tweet_text = st.text_area('Input Text')
	if st.button('Analyze'):
		text = preprocessing(tweet_text)
		result = get_prediction(text)
		write_result(st, tweet_text, text, result)

with st.expander('Upload File (CSV)'):
	file = st.file_uploader('Upload File', type=['csv'])
	delim = st.text_input('Delimiter', value=',', max_chars=1)
	column = st.text_input('Column Name', value='text')
	if st.button('Analyze', key='upload'):
		handle_uploaded_file(st, file, delim, column)