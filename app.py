from flask import Flask, request, jsonify
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
#Import necessary libraries

#Initialize Flask application
app = Flask(__name__)

#Load the pre-trained SentenceTransformer model
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

#Function to compare similarity between two texts
def comparison(text1, text2):
    #Encode the text2 sentence
    test_vec = model.encode(text2)
    
    #Encode the text1 sentence
    sentence_vec = model.encode(text1)
    
    #Calculate cosine similarity
    similarity_score = 1 - distance.cosine(test_vec, sentence_vec)
    
    #Ensure similarity score is bounded within range
    similarity_score = max(0, min(similarity_score, 1))
    
    return similarity_score

#Define a route to handle POST requests for calculating similarity
@app.route('/calculate_similarity', methods=['POST'])

def calculate_similarity():
    #Extract data from JSON request
    data = request.get_json()
    text1 = data.get('text1', '')
    text2 = data.get('text2', '')

    #Call the comparison function to calculate similarity score
    similarity_score = comparison(text1, text2)

    #Prepare response
    response = {'similarity score': similarity_score}
    
    #Return response as JSON
    return jsonify(response)

#Run the Flask app
if __name__ == '__main__':
    # Run the Flask app on specified host and port for testing
    app.run(host='0.0.0.0', port=80)
