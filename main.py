import pickle

with open('./dataset/model/sental_model.pkl', 'rb') as f:
		data = pickle.load(f)
with open('./dataset/model/vectorizer.pkl', 'rb') as f:
		vectorizer = pickle.load(f)

def predict(text):
		polarity_decode = {0 : 'Negative', 1 : 'Positive'}

		vector = vectorizer.transform([text])
		polarity = data.predict(vector)[0]
		sentiment = polarity_decode[polarity]