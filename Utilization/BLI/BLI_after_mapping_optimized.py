import pandas as pd
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. 데이터 준비
df = pd.read_csv("./translated_csv_file_optimized.csv")

# 2. 임베딩 로드
ko_model = KeyedVectors.load_word2vec_format('./data/cc.ko.300.vec', binary=False)
es_model = KeyedVectors.load_word2vec_format('./data/cc.es.300.vec', binary=False)

# 3. 유사도 계산 함수
def calculate_similarity(ko_word, es_word):
    if ko_word in ko_model.key_to_index and es_word in es_model.key_to_index:
        ko_embedding = ko_model[ko_word]
        es_embedding = es_model[es_word]
        similarity = cosine_similarity([ko_embedding], [es_embedding])
        return similarity[0][0]
    else:
        return None

# 4. 성능 평가
df['similarity'] = df.apply(lambda row: calculate_similarity(row['Korean'], row['Spanish_Translated']), axis=1)

# 상위 N개 단어 쌍의 평균 유사도 출력
N = 100  # 분석할 단어 쌍의 수
top_similarities = df['similarity'].dropna().nlargest(N).mean()
print(f"Top {N} word pairs' average similarity: {top_similarities}")
