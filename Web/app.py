from flask import Flask, render_template, request, jsonify
import pandas as pd
import random
from sentence_transformers import SentenceTransformer, util
import torch

app = Flask(__name__)

# Load your CSV data
df = pd.read_csv('test.csv')

# Load the sentence-transformers model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/korean_to_spanish')
def korean_to_spanish():
    return render_template('index_korean_to_spanish.html')

@app.route('/spanish_to_korean')
def spanish_to_korean():
    return render_template('index_spanish_to_korean.html')


@app.route('/get_random_sentence/<language>')
def get_random_sentence(language):
    if language not in ['ko', 'es']:
        return jsonify(error="Invalid language"), 400  # 잘못된 언어 값에 대한 오류 응답 반환
    if language == 'ko':
        sentence = random.choice(df['Korean'].tolist())
    else:
        sentence = random.choice(df['Spanish'].tolist())
    return jsonify(sentence=sentence)


@app.route('/calculate_similarity', methods=['POST'])
def calculate_similarity_endpoint():
    data = request.json
    input_sentence = data['input_sentence']
    target_sentence = data['target_sentence']

    # Calculate embeddings
    input_embedding = model.encode(input_sentence, convert_to_tensor=True)
    target_embedding = model.encode(target_sentence, convert_to_tensor=True)

    # Calculate cosine similarity
    similarity_score = util.pytorch_cos_sim(input_embedding, target_embedding)

    # Convert similarity score to a scalar for easy JSON serialization
    similarity_score = similarity_score.item()

    # Prepare the response data including the original sentence
    response_data = {
        'input_sentence': input_sentence,
        'target_sentence': target_sentence,
        'similarity_score': similarity_score
    }

    return jsonify(response_data)




if __name__ == '__main__':
    app.run(debug=True)
