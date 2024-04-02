from flask import Flask, request, jsonify
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
import numpy as np 
import pandas as pd
nltk.download('punkt')
app = Flask(__name__)

# Your model import and initialization here
# For demonstration, let's assume a function called calculate_similarity_score(text1, text2) is defined elsewhere
def model(text1, text2):
        # Tokenizing the data
	tokenized_data = [word_tokenize(text1.lower())]

	# Creating TaggedDocument objects
	tagged_data = [TaggedDocument(words=words, tags=[str(idx)])
				for idx, words in enumerate(tokenized_data)]


	# Training the Doc2Vec model
	model = Doc2Vec(vector_size=100, window=1, min_count=1, workers=4, epochs=1000)
	model.build_vocab(tagged_data)
	model.train(tagged_data, total_examples=model.corpus_count,
				epochs=model.epochs)

	# Infer vector for a new document
	new_document = text2
	#print('Original Document:', new_document)

	inferred_vector = model.infer_vector(word_tokenize(new_document.lower()))

	# Find most similar documents
	similar_documents = model.dv.most_similar(
		[inferred_vector], topn=len(model.dv))

	# Print the most similar documents
	for index, score in similar_documents:
		return score
@app.route('/calculate_similarity', methods=['POST'])
def calculate_similarity():
    data = request.get_json()
    text1 = data.get('text1', '')
    text2 = data.get('text2', '')

    # Call your model function to calculate similarity score
    similarity_score = model(text1,text2)

    response = {'similarity score': abs(similarity_score)}
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
