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

float_to_percent = lambda x: f"{x*100:.1f}%"

def get_prediction(tweet_text):
	model = load_model()
	vectorizer = load_vectorizer()
	tweet_vector = vectorizer.transform([tweet_text])
	subjectivity = model.predict(tweet_vector)
	probabilities = model.predict_proba(tweet_vector)
	return subjectivity, probabilities

def write_result(st, text_ori, text_clean, result):
	subjectivity, probabilities = result
	negative_proba = probabilities[0][0]
	positive_proba = probabilities[0][1]
	sentiment = 'Negative' if subjectivity[0] == 0 else 'Positive'
	
	st.info(f"""
	Text Preprocessing
	- Original Text: `{text_ori}`
	- Preprocessed Text: `{text_clean}`
	""")
	
	st.info(f"""
	Polarity score
	- Negative score: `{negative_proba}` ({float_to_percent(negative_proba)})
	- Positive score: `{positive_proba}` ({float_to_percent(positive_proba)})
	""")

	if sentiment == 'Negative':
		st.error(f"Sentiment: `{sentiment}`")
	else:
		st.success(f"Sentiment: `{sentiment}`")

st.title("Sentiment Analysis of Tweets about Indonesia's Fuel Price Hike")
st.write("This is a simple sentiment analysis of tweets about Indonesia's fuel price hike developed by [Davi Nomoeh Dani](https://mr687.github.io). The model is trained using a SVM classifier.")

with st.expander('Manual Input'):
	tweet_text = st.text_area('Input Text')
	if st.button('Analyze'):
		text = preprocess_text(tweet_text)
		result = get_prediction(text)
		write_result(st, tweet_text, text, result)

# with st.expander('Upload File (CSV)'):
# 	file = st.file_uploader('Upload File', type=['csv'])
# 	delim = st.text_input('Delimiter', value=',', max_chars=1)
# 	column = st.text_input('Column Name', value='text')
# 	if st.button('Analyze', key='upload'):
# 		df = pd.read_csv(file, delimiter=delim)
# 		data_count = len(df)

# 		with st.spinner('Wait for it...'):
# 			progress_bar = st.progress(0)
# 			df['text_clean'] = df[column].swifter.apply(preprocess_text)
# 			progress_bar.progress(data_count * 0.5)
# 			df['sentiment'] = df['text_clean'].swifter.apply(get_prediction)
# 			progress_bar.progress(100)
# 			st.write(df[[column, 'sentiment']])
		
# 		st.balloons()

# 		st.download_button(
# 			label='Download Result',
# 			data=df.to_csv(index=False).encode('utf-8'),
# 			file_name=f'{file.name}-result.csv',
# 			mime='text/csv'
# 		)