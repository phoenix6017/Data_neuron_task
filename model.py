#Import necessary libraries
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
import nltk
import numpy as np 
import pandas as pd

#Read data from CSV file
data = pd.read_csv(r"C:\Users\Acer\Dropbox\My PC (LAPTOP-91U6NI9O)\Desktop\Comding\Data_Neuron\DataNeuron_Text_Similarity.csv")

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')

# Assuming 'data' is your DataFrame containing 'text1' and 'text2' columns
for i, row in data.iterrows():
    # Sample sentence
    sentence = data['text1'][i]
    test = data['text2'][i]
    
    print('Test sentence:', test)
    
    # Encode the test sentence
    test_vec = model.encode(test)
    
    # Encode the sample sentence
    sentence_vec = model.encode(sentence)
    
    # Calculate cosine similarity
    similarity_score = 1 - distance.cosine(test_vec, sentence_vec)
    
    # Ensure similarity score is bound between 0 and 1
    similarity_score = max(0, min(similarity_score, 1))
    
    print(f'For {i+1}\nSimilarity Score = {similarity_score}')