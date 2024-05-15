
from gensim.models import KeyedVectors

# 한국어 FastText 벡터 로딩
ko_model_path = './data/cc.ko.300.vec'  # 실제 파일 경로로 수정
ko_model = KeyedVectors.load_word2vec_format(ko_model_path, binary=False, encoding='utf-8')

# 스페인어 FastText 벡터 로딩
es_model_path = './data/cc.es.300.vec'  # 실제 파일 경로로 수정
es_model = KeyedVectors.load_word2vec_format(es_model_path, binary=False, encoding='utf-8')

# 한국어 모델 저장
ko_model.save('ko_model.bin')

# 스페인어 모델 저장
es_model.save('es_model.bin')