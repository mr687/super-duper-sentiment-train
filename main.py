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

def write_result_sentiment(st, result):
	if result == 'Negative':
		st.error(f"Sentiment: {result}")
	else:
		st.success(f"Sentiment: {result}")

def write_result(st, text_ori, text_clean, result):
	st.write(f"Tweet Original: `{text_ori}`")
	st.write(f"Preprocessed Tweet: `{text_clean}`")
	write_result_sentiment(st, result)

st.title("Sentiment Analysis of Tweets about Indonesia's Fuel Price Hike")
st.write("This is a simple sentiment analysis of tweets about Indonesia's fuel price hike developed by [Davi Nomoeh Dani](https://mr687.github.io). The model is trained using a SVM classifier.")

with st.expander('Example Tweets', expanded=True):
	samples = [
		"pertamina naikin harga bensin lagi, kapan ya harga bensin naik lagi?",
		"@pertamina emang gak ada yg bisa diandalkan, kalo gak ada yg bisa diandalkan, kenapa kita harus bayar pajak? #HargaPertaminaMakinMahal",
	]
	for i, sample in enumerate(samples):
		text = preprocess_text(sample)
		result = get_prediction(text)
		write_result(st, sample, text, result)
		if i < len(samples) - 1:
			st.write('---')


with st.expander('Manual Input'):
	tweet_text = st.text_area('Tweet Text')
	if st.button('Analyze'):
		text = preprocess_text(tweet_text)
		result = get_prediction(text)
		write_result(st, tweet_text, text, result)
		st.balloons()

with st.expander('Upload File (CSV)'):
	file = st.file_uploader('Upload File', type=['csv'])
	delim = st.text_input('Delimiter', value=',', max_chars=1)
	column = st.text_input('Column Name', value='text')
	if st.button('Analyze', key='upload'):
		df = pd.read_csv(file, delimiter=delim)
		data_count = len(df)

		with st.spinner('Wait for it...'):
			progress_bar = st.progress(0)
			df['text_clean'] = df[column].swifter.apply(preprocess_text)
			progress_bar.progress(data_count * 0.5)
			df['sentiment'] = df['text_clean'].swifter.apply(get_prediction)
			progress_bar.progress(100)
			st.write(df[[column, 'sentiment']])
		
		st.balloons()

		st.download_button(
			label='Download Result',
			data=df.to_csv(index=False).encode('utf-8'),
			file_name=f'{file.name}-result.csv',
			mime='text/csv'
		)