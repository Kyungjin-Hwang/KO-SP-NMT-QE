import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE

# FastText 모델 로드
ko_model = KeyedVectors.load_word2vec_format('./data/cc.ko.300.vec', binary=False)
es_model = KeyedVectors.load_word2vec_format('./data/cc.es.300.vec', binary=False)

# 번역된 단어 쌍 로드
df_translated = pd.read_csv("./translated_csv_file_optimized.csv")

# 시각화할 단어 쌍 선택 (예: 상위 N개)
N = 100
top_pairs = df_translated.head(N)

# 임베딩 추출
ko_embeddings = np.array([ko_model[word] for word in top_pairs['Korean'] if word in ko_model])
es_embeddings = np.array([es_model[word] for word in top_pairs['Spanish_Translated'] if word in es_model])

# t-SNE를 사용하여 임베딩을 2차원으로 축소
tsne = TSNE(n_components=2, random_state=0)
reduced_embeddings = tsne.fit_transform(np.vstack((ko_embeddings, es_embeddings)))

# 시각화
fig, ax = plt.subplots()
ko_points = reduced_embeddings[:len(ko_embeddings)]
es_points = reduced_embeddings[len(ko_embeddings):]

ax.scatter(ko_points[:, 0], ko_points[:, 1], color='blue', label='Korean')
ax.scatter(es_points[:, 0], es_points[:, 1], color='red', label='Spanish')
ax.legend()

plt.show()
