import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import pandas as pd

# Assuming mappings are stored in a separate directory
ko_en_mapping_path = './src/muse_output/best_mapping_ko_en.pth'
en_es_mapping_path = './src/muse_output_en-es/best_mapping_en_es.pth'
"""
import torch
d = torch.load(ko_en_mapping_path)
print(d)

# Load the mapping file
# Print the type and content (if it's not too large) to understand its structure
print(type(ko_en_mapping_path))
if isinstance(ko_en_mapping_path, dict):
    print(ko_en_mapping_path.keys())
"""
# 매핑 파일 로드 함수
def load_mapping(path):
    # Load the entire file content
    mapping = torch.load(path)
    # If the file directly contains a numpy array or a PyTorch tensor, convert it to numpy array
    if isinstance(mapping, dict) and 'weight' in mapping:
        # If the mapping is stored in a dictionary under the 'weight' key
        return mapping['weight'].cpu().numpy()
    elif torch.is_tensor(mapping):
        # If the mapping is a PyTorch tensor
        return mapping.cpu().numpy()
    else:
        # Directly return the numpy array or raise an error if the format is unexpected
        raise ValueError("The mapping file format is not recognized. Please check the file content.")

# Load mappings
ko_en_mapping = torch.load(ko_en_mapping_path)
en_es_mapping = torch.load(en_es_mapping_path)

# Load FastText models
en_model = KeyedVectors.load_word2vec_format('./data/cc.en.300.vec', binary=False)
es_model = KeyedVectors.load_word2vec_format('./data/cc.es.300.vec', binary=False)

original_file_path = './data/cc.ko.300.vec'
utf8_file_path = './data/cc.ko.300.vec.utf8'

# 원본 파일을 줄 단위로 읽어서 새 파일에 UTF-8로 인코딩하여 저장
with open('./data/cc.ko.300.vec', 'r', encoding='latin1') as original_file, \
     open('./data/cc.ko.300.vec.utf8', 'w', encoding='utf-8') as utf8_file:
    for line in original_file:
        utf8_file.write(line)


# 변환된 파일 로드
ko_model = KeyedVectors.load_word2vec_format(utf8_file_path, binary=False)
en_model = KeyedVectors.load_word2vec_format('./data/cc.en.300.vec', binary=False)
es_model = KeyedVectors.load_word2vec_format('./data/cc.es.300.vec', binary=False)

# Updated get_embedding function with error handling
def get_embedding(word, model):
    try:
        if word in model.key_to_index:  # gensim 4.0.0 updates 'vocab' to 'key_to_index'
            return model[word]
        else:
            return np.zeros(model.vector_size)
    except TypeError:
        print(f"Error with word: {word}")
        return np.zeros(model.vector_size)

# Find closest word based on cosine similarity
def find_closest_word(embedding, model):
    similarities = cosine_similarity([embedding], model.vectors)[0]
    closest_word_idx = similarities.argmax()
    return model.index_to_key[closest_word_idx]

# Translation function
# Updated translation function to reflect changes
def translate_ko_to_es(ko_word, ko_model, en_model, es_model, ko_en_mapping, en_es_mapping):
    ko_embedding = get_embedding(ko_word, ko_model)
    en_embedding = np.dot(ko_embedding, ko_en_mapping)  # Korean to English mapping
    en_word = find_closest_word(en_embedding, en_model)
    es_embedding = np.dot(get_embedding(en_word, en_model), en_es_mapping)  # English to Spanish mapping
    es_word = find_closest_word(es_embedding, es_model)
    return es_word

# Load CSV, translate, and save
df = pd.read_csv("./Dataset_for_NLP.csv")

# Ensure all Korean words are strings
df['Korean'] = df['Korean'].astype(str)
# Apply the translation function
df['Spanish_Translated'] = df['Korean'].apply(lambda ko_word: translate_ko_to_es(ko_word, ko_model, en_model, es_model, ko_en_mapping, en_es_mapping))

# Save the results
df.to_csv("./translated_csv_file.csv", index=False)
