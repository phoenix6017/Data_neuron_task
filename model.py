from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
import numpy as np 
import pandas as pd
nltk.download('punkt')

# Sample data
data=pd.read_csv(r"C:\Users\Acer\Dropbox\My PC (LAPTOP-91U6NI9O)\Desktop\Comding\Data_Neuron\DataNeuron_Text_Similarity.csv")
for i,row in data.iterrows():
	# Tokenizing the data
	tokenized_data = [word_tokenize(data['text1'][i].lower())]

	# Creating TaggedDocument objects
	tagged_data = [TaggedDocument(words=words, tags=[str(idx)])
				for idx, words in enumerate(tokenized_data)]


	# Training the Doc2Vec model
	model = Doc2Vec(vector_size=100, window=1, min_count=1, workers=4, epochs=1000)
	model.build_vocab(tagged_data)
	model.train(tagged_data, total_examples=model.corpus_count,
				epochs=model.epochs)

	# Infer vector for a new document
	new_document = data['text2'][i]
	#print('Original Document:', new_document)

	inferred_vector = model.infer_vector(word_tokenize(new_document.lower()))

	# Find most similar documents
	similar_documents = model.dv.most_similar(
		[inferred_vector], topn=len(model.dv))

	# Print the most similar documents
	for index, score in similar_documents:
		print(f"Document {i}: Similarity Score: {score}")
