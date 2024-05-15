import torch
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool
import pandas as pd

def load_mapping(path):
    # 텐서를 로드하고 NumPy 배열로 변환합니다.
    mapping_tensor = torch.load(path)
    if isinstance(mapping_tensor, np.ndarray):
        return mapping_tensor
    elif torch.is_tensor(mapping_tensor):
        return mapping_tensor.cpu().numpy()
    else:
        raise TypeError("Loaded mapping is neither a PyTorch tensor nor a numpy array.")

def get_embedding(word, model):
    # 주어진 단어에 대한 임베딩을 반환합니다. 단어가 모델에 없으면 0 벡터를 반환합니다.
    try:
        return model[word]
    except KeyError:
        return np.zeros(model.vector_size)

def find_closest_word(embedding, model):
    # 임베딩에 가장 가까운 단어를 찾습니다.
    similarities = cosine_similarity([embedding], model.vectors)[0]
    return model.index_to_key[similarities.argmax()]

def translate_ko_to_es(ko_word, ko_model, en_model, es_model, ko_en_mapping, en_es_mapping):
    # 한국어 단어를 스페인어 단어로 번역합니다.
    ko_embedding = get_embedding(ko_word, ko_model)
    en_embedding = np.dot(ko_embedding, ko_en_mapping)
    en_word = find_closest_word(en_embedding, en_model)
    es_embedding = np.dot(en_embedding, en_es_mapping)
    es_word = find_closest_word(es_embedding, es_model)
    return es_word

def translate_batch(batch_data):
    # 배치 데이터에 대해 번역을 수행하는 함수입니다.
    ko_words, ko_model, en_model, es_model, ko_en_mapping, en_es_mapping = batch_data
    translated_words = [translate_ko_to_es(word, ko_model, en_model, es_model, ko_en_mapping, en_es_mapping) for word in ko_words]
    return translated_words

# 모델과 매핑을 로드합니다.
ko_model = KeyedVectors.load_word2vec_format('./data/cc.ko.300.vec.utf8', binary=False)
en_model = KeyedVectors.load_word2vec_format('./data/cc.en.300.vec', binary=False)
es_model = KeyedVectors.load_word2vec_format('./data/cc.es.300.vec', binary=False)
ko_en_mapping = load_mapping('./src/muse_output/best_mapping_ko_en.pth')
en_es_mapping = load_mapping('./src/muse_output_en-es/best_mapping_en_es.pth')

# CSV 파일 로드
df = pd.read_csv("./Dataset_for_NLP.csv")

# 병렬 처리를 위해 데이터를 배치로 나눕니다.
num_batches = 10  # 예시로 10개의 배치로 나눕니다.
batch_size = len(df) // num_batches
batches = [(df['Korean'][i:i + batch_size].tolist(), ko_model, en_model, es_model, ko_en_mapping, en_es_mapping) for i in range(0, len(df), batch_size)]

# 병렬 처리 실행
with Pool(processes=4) as pool:  # 프로세스 수는 시스템에 따라 조정
    results = pool.map(translate_batch, batches)

# 결과 병합 및 저장
df['Spanish_Translated'] = np.concatenate(results)
df.to_csv("./translated_csv_file_optimized.csv", index=False)
