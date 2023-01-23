import nltk
import pandas as pd
import re
import string
import numpy as np
import os
import pickle as pkl

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import Counter

root_path = os.path.dirname(os.path.abspath(__file__))

def get_filepath(filename):
	return os.path.join(root_path, filename)

def case_folding(text):
	return text.lower()

def remove_repetitive_char(tokens):
	execption = ['taat', 'saat']
	result = []
	for token in tokens:
		if token in execption:
			result.append(token)
		else:
			result.append(re.sub(r'(.)\1{1,}', r'\1', token))
	return result

def load_term_dict():
	term_dict_path = get_filepath('dataset/processed/processed_terms_dict.pkl')
	try:
		with open(term_dict_path, 'rb') as f:
			term_dict = pkl.load(f)
		print(f"Term dictionary: {len(term_dict)} loaded")
		return term_dict
	except FileNotFoundError:
		return {}

def save_term_dict(term_dict):
	term_dict_path = get_filepath('dataset/processed/processed_terms_dict.pkl')
	with open(term_dict_path, 'wb') as f:
		pkl.dump(term_dict, f)

def clean_tweet(tweet):
	# Remove @mentions
	tweet = re.sub(r'@\S+', ' ', tweet)
	# Remove hashtags
	tweet = re.sub(r'#\S+', ' ', tweet)
	# Remove URLs
	tweet = re.sub(r'https?://[A-Za-z0-9./]+', ' ', tweet)
	# Remove RT
	tweet = re.sub(r'RT : ', ' ', tweet)
	# Remove numbers
	tweet = re.sub(r'[0-9]', ' ', tweet)
	# Remove non-ASCII characters
	tweet = tweet.encode('ascii', 'ignore').decode('ascii')
	# Remove punctuation
	# tweet = re.sub(r'[^\w\s]', ' ', tweet)
	tweet = tweet.translate(str.maketrans('', '', string.punctuation))
	# Remove whitespace
	tweet = re.sub(r'\s+', ' ', tweet)
	# Remove leading and trailing whitespace
	tweet = tweet.strip()
	# Keep tweet with more than 1 characters
	tweet = ' '.join([w for w in tweet.split() if len(w) > 2])
	return tweet

def tokenize(tweet):
	return nltk.word_tokenize(tweet)

def load_normalization_list():
	normalization_list_path = get_filepath('dataset/wordlist/normalization_list.csv')
	normalization_list = pd.read_csv(normalization_list_path, delimiter=',')
	list_normalize_targets = list(normalization_list['target'])
	list_normalize_replacements = list(normalization_list['replacement'])
	print(f"Normalization list: {len(list_normalize_targets)}")
	return list_normalize_targets, list_normalize_replacements

def normalize(tokens, targets, replacements):
	if len(targets) != len(replacements):
		raise ValueError('Targets and replacements must have the same length')
	result = []
	for token in tokens:
		if token in targets:
			replacement = replacements[targets.index(token)]
			if replacement != '' and replacement is not None and replacement != ' ' and replacement is not np.nan and replacement != 'nan' and replacement != 'NaN' and replacement != 'NAN':
				result.append(replacement)
		else:
			result.append(token)
	return result

def load_stopwords_list():
	indonesian_stopwords_list = nltk.corpus.stopwords.words('indonesian')
	english_stopwords_list = nltk.corpus.stopwords.words('english')
	custom_stopwords_list = ['aaaaaaaaa', 'aaaaaaaaaah', 'aaaahhh', 'aaahhh', 'aah', 'aatu', 'ngaoahahahhahahaha',
													'wkwkwk', 'slebew', 'sih', 'nih', 'deh', 'nya', 'mah', 'breaking', 'news', 'wkwkw']

	lexicon_all = pd.read_csv(get_filepath('dataset/wordlist/lexicon_dict_all.csv'), delimiter=',')
	lexicon_all = list(lexicon_all['word'])

	indonesian_stopwords_list = [word for word in indonesian_stopwords_list if word not in lexicon_all]

	list_stopwords = indonesian_stopwords_list + english_stopwords_list + custom_stopwords_list
	print(f"Indonesian stopwords: {len(indonesian_stopwords_list)}")
	print(f"English stopwords: {len(english_stopwords_list)}")
	print(f"Custom stopwords: {len(custom_stopwords_list)}")
	print(f"Total stopwords: {len(list_stopwords)}")
	return list_stopwords

def remove_stopwords(tokens, stopwords):
	return [token for token in tokens if token not in stopwords]

def build_stemmer():
	stem_factory = StemmerFactory()
	stemmer = stem_factory.create_stemmer()
	return stemmer

stemmer = build_stemmer()
terms_dict = load_term_dict()
def stemming_text(text):
	return stemmer.stem(f'{text}')
def do_stemming(tokens):
	return [terms_dict[token] for token in tokens]
def prepare_stemming(tokens):
	for token in tokens:
		if token not in terms_dict:
			terms_dict[token] = stemming_text(text=token)
	return tokens
def stemming(df, column='tokens'):
	print('Stemming...')
	df[column] = df[column].swifter.apply(prepare_stemming)
	df[column] = df[column].swifter.apply(do_stemming)
	df[column] = df[column].swifter.apply(remove_stopwords, args=(list_stopwords,))
	df['review'] = df[column].swifter.apply(untokenizer)

	save_term_dict(term_dict=terms_dict)
	return df

def clean_data_with_rare_word(df, column='tokens', threshold=2):
	print('Cleaning data with rare word...')
	print(f"Before: {df.shape}")
	token_counts = Counter([token for tokens in df[column] for token in tokens])
	filtered_words = [token for token, count in token_counts.items() if count >= threshold]
	df = df[df[column].apply(lambda x: any(token in filtered_words for token in x))]
	df = df.reset_index(drop=True)
	print(f"After: {df.shape}")
	return df

def clean_duplicate_data(df, column='tokens'):
	print('Cleaning duplicate data...')
	print(f"Before: {df.shape}")
	df = df.drop_duplicates(subset=column, keep="first")
	df = df.dropna(subset=[column])
	df = df[df[column].map(len) > 2]
	print(f"After: {df.shape}")
	return df

def clean_data(df, column='tokens',threshold=2):
	df = clean_data_with_rare_word(df, column, threshold)
	df = clean_duplicate_data(df, column)
	df[column] = df[column].swifter.apply(remove_repetitive_char)
	return df

def untokenizer(tokens):
	return ' '.join(tokens)

list_normalize_targets, list_normalize_replacements = load_normalization_list()
list_stopwords = load_stopwords_list()
def preprocessing_dataframe(df, column='content'):
	df['text_clean'] = df[column].swifter.apply(clean_tweet)
	df['case_folding'] = df['text_clean'].swifter.apply(case_folding)
	df['tokens'] = df['case_folding'].swifter.apply(tokenize)
	df['review'] = df['tokens'].swifter.apply(untokenizer)
	return df

def preprocessing_text(text):
	text = clean_tweet(text)
	text = case_folding(text)
	tokens = tokenize(text)
	tokens = normalize(tokens, list_normalize_targets, list_normalize_replacements)
	tokens = remove_stopwords(tokens, list_stopwords)
	tokens = remove_repetitive_char(tokens)
	tokens = [stemmer.stem(token) for token in tokens]
	return ' '.join(tokens)

def preprocessing(df_or_text, column=None):
	if isinstance(df_or_text, pd.DataFrame):
		return preprocessing_dataframe(df_or_text, column)
	return preprocessing_text(df_or_text)

def normalizer(df, column='tokens'):
	df[column] = df[column].swifter.apply(normalize, args=(list_normalize_targets, list_normalize_replacements))
	df[column] = df[column].swifter.apply(remove_stopwords, args=(list_stopwords,))
	df['review'] = df[column].swifter.apply(untokenizer)
	return df