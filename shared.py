import nltk
import pandas as pd
import re
import string
import numpy as np

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

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
	normalization_list_path = './dataset/wordlist/normalization_list.csv'
	normalization_list = pd.read_csv(normalization_list_path, delimiter=',')
	list_normalize_targets = list(normalization_list['target'])
	list_normalize_replacements = list(normalization_list['replacement'])
	print(f"Normalization list: {len(list_normalize_targets)}")
	return list_normalize_targets, list_normalize_replacements

def normalize(tokens, targets, replacements):
	return tokens
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

	lexicon_all = pd.read_csv('./dataset/wordlist/lexicon_dict_all.csv', delimiter=',')
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

def remove_token_with_recurring_characters(tokens):
	return [token for token in tokens if not re.search(r'(.)\1{3,}', token)]

def build_stemmer():
	stem_factory = StemmerFactory()
	stemmer = stem_factory.create_stemmer()
	return stemmer

stemmer = build_stemmer()
def stemming(tokens_bag):
	terms_dict = {}
	for tokens in tokens_bag:
		for token in tokens:
			if token not in terms_dict:
				terms_dict[token] = ''

	print(f"Unique terms: {len(terms_dict)}")

	for i, term in enumerate(terms_dict):
		terms_dict[term] = stemmer.stem(f'{term}')
		if i % 1000 == 0:
			print(f"On processing... {i} terms have been stemmed")
	return terms_dict

def stemming_text(text):
	return stemmer.stem(text)

def preprocess_text(text):
	stemmer = build_stemmer()

	text = case_folding(text)
	text = clean_tweet(text)
	tokens = tokenize(text)
	list_normalize_targets, list_normalize_replacements = load_normalization_list()
	tokens = normalize(tokens, list_normalize_targets, list_normalize_replacements)
	list_stopwords = load_stopwords_list()
	tokens = remove_stopwords(tokens, list_stopwords)
	tokens = [stemmer.stem(token) for token in tokens]
	tokens = remove_token_with_recurring_characters(tokens)
	return ' '.join(tokens)