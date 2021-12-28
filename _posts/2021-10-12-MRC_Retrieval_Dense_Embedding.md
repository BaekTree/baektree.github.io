---
title: "MRC_Retrieval_Dense_Embedding"
last_modified_at: 2021-10-12T16:20:02-05:00
categories:
  - nlp
  - boostcamp
tags:
  - MRC
  - Retrieval
  - blog
---
# Dense Embedding Retrieval

sparse embedding의 단점
passage embedding: 문서를 벡터로 변환하는 것. sparse 벡터는 bag of words을 사용해서... 문서에 안나오는 vocab단어는  tfidf 값이 0이 되어서 그 위치에서 벡터가 0이 된다.

하지만 더 큰 문제점은... 유사성을 고려하지 못한다. 그냥 중요도 기반에서 벡터를 구성하기 때문이다. 

dense embedding은 word2vec과 같이... 유사성을 가질 수 있음.

더 작은 차원의 고밀도로 만들 수 있음. latent feature에 매핑. 

비교
sparse
장점
term이 정확이 일치해야 하는 경우 성능 좋음
단점
임베딩 구축되고 나서 추가 학습 불가능

dense
유사성과 맥락 파악이 필요할 때 성능이 좋다
추가적인 학습이 가능

사전학습에 좋음. neural net 사용. accuracy 많이 올라감. 

Retrieval에 활용
질문을 인코더(버트)에 넣는다. CLS을 뽑음. 이게 embedding된 벡터. 차원 변환해서 hq으로 만든다. 

각 문서에서도 인코더(버트) 넣어서 CLS 뽑아서 차원변환해서 hb만든다. 

보통 q와 passage에 다른 인코더를 쓴다. 

문서 벡터와 q 벡터와 유사도 계산한다. cosine sim = dot prod으로 가장 유사한 문서 뽑아와서 MRC에 넣는다. 

버트에 q와 passage을 독립적인 두개의 버트에 넣는다. 가각 넣는다. 같은 모델인지 다른 모델인지는 디자인 초이스.

정답인 passage는 유사도가 1에 가까워야 하고, 정답이 아는 passage는 유사도가 -1에 가까워야 한다.

학습 목표: 연관된 question 과 passage 사이의 sim을 높이는 것(loss가 cosine sim 등이 될것이다!)

challenge: 연관된 q와 passage들의 pair을 어떻게 찾는가? 어떤 기준으로 연관된 것인지 아닌지를 판단해서 데이터를 training set을 만드는가? 여러 방법이 있지만 가장 쉬운 것: 기존 MRC 데이터셋을 활용. MRC의 정답 passage 페어가 정답. 다른 passage는 틀린 답.

negative examples 뽑기
corpus 내에서 랜덤하게 뽑기
높은 tf-idf을 가지지만 답을 포함하지 않는 샘플. 즉 실제 정답과 유사한 문장이지만 답이 아닌 passage들.

objective function
positive passage에 대한 NLL loss. 

전체 문서들 중에서 ground truth가 가장 점수가 높아야 하게 하는 cross entropy와 동일.

metric: retrieve 된 top k 중에 답을 포함하는 passage의 비율. EM으로 측정할 것 같은데? 

dense embedding 만드는 dense encoder

학습 방법 개선(DPR): 지금과 같은 sim 유사도를 넘어서 더 복잡한 모델을 말하는 듯?
인코더 모델 개선(BERT 보다 큰 모델): 현재 모델 구조를 사용하되 인코더 모델을 바꾸는 듯?
데이터 개선(전처리, augmentaiton, 데이터 추가)
