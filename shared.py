import nltk
import pandas as pd
import re

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def case_folding(text):
	return text.lower()

def clean_tweet(tweet):
	# Remove @mentions
	tweet = re.sub(r'@\S+', ' ', tweet)
	# Remove URLs
	tweet = re.sub(r'https?://[A-Za-z0-9./]+', ' ', tweet)
	# Remove RT
	tweet = re.sub(r'RT : ', ' ', tweet)
	# Remove punctuation
	tweet = re.sub(r'[^\w\s]', ' ', tweet)
	# Remove numbers
	tweet = re.sub(r'[0-9]', ' ', tweet)
	# Remove whitespace
	tweet = re.sub(r'\s+', ' ', tweet)
	# Remove leading and trailing whitespace
	tweet = tweet.strip()
	# Remove non-ASCII characters
	tweet = tweet.encode('ascii', 'ignore').decode('ascii')
	# Keep tweet with more than 2 characters
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
	return [replacements[targets.index(token)] if token in targets else token for token in tokens]

def load_stopwords_list():
	indonesian_stopwords_list = nltk.corpus.stopwords.words('indonesian')
	english_stopwords_list = nltk.corpus.stopwords.words('english')
	custom_stopwords_list = ['pertamina', 'vivo']

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

def stemming(tokens_bag):
	stemmer = build_stemmer()

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