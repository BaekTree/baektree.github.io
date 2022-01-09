---
title: "attention-is-not-explanation-review"
last_modified_at: 2021-10-12T16:20:02-05:00
categories:
  - nlp
  - paper-review
tag:
  - transformer
  - paper
  - NLP
---
## 직관적인 내용 정리

## 배경: 
NLP에서 attention 쓰는 것이 SOTA을 찍으면서 이제는 아주 당연하게 사용하는 것이 됨. 그리고 모델의 예측하는 과정을 attention weights의 값을 통해서 설명하려는 경우도 아주 아주 많아짐. 문맥에서 이 단어와 저 단어가 높은 attention weight을 가져서 이런 결과를 예측했다. 혹은 예측 값에 대해서 intput 들에 attention weight을 살펴보니 부정적인 단어에 값이 커서 negative sentiment을 분류했더라 등등.

## 이 논문의 주장: 
이렇게 attention weight을 가지고 prediction을 설명하려는 것이 타당하지 않음. 우리가 실험을 해봤는데, 이 attention weight 값이 그냥 (과장 좀 붙여서) 랜덤임! 이걸로 predictoin을 설명하는건 그냥 스토리 짜맞추기임... 

## 실험을 하면서 알게된 점

* 실험1: attention weight랑 input의 feature importance의 correlation을 살펴봤는데 두개가 유의미한 관련이 없었음. 

* 실험2: 그래서 혹시 기존의 attention 방식이 문제인가 싶어서 다른 방식의 attention weight을 만들어봤거든? 근데 얘들도 input feature importance랑 마찬가지로 크게 correlation 연관이 없었음... 

* 실험 준비
  * task: sentiment classification하는 task. positive or negative 구분.
  * 모델: attention이 있는 bi-RNN으로 seq2seq의 인코더 디코더 모델 씀. input은 one-hot vector으로 했음. 그리고 각 time step에서 hidden state vector 마다 linear 레이어 → relu을 걸어서 average embedding으로 변형한 모델을 씀(사실 몬 소린지 정확히 모르겠음. 각 time step 마다 다음 step 으로 넘어가기 전에 이 과정을 거친다음에 다음 step으로 보냈다는 말인가? 근데 왜 이런 과정을 거쳤지? 이 설명은 안나옴...). attention은 각 time step에서의 hidden state가 Q가 되고 나머지 step들의 hidden state가 K가 되어서 유사도를 구하고, 각 step의 hidden state가 v와 내적(그 안에 학습 weight 몇개 넣음) 함. 그리고 scaled dot product까지 해서 attention weight을 만든다(일반적인 틀내스포머랑 아주 유사하게 설계했음). 이렇게 누적시켜서 디코더로 보냄. 디코더는, 인코더의 attention과 hidden state을 내적한 값을 받아서(마침내 문맥이 고려되어 있는 hidden representation) sigmoid 받아서 positive or negative sentiment 구분 함. 그냥 디코더는 binary classification만 하는 경우인 것 같다.

### 실험1: attention과 input의 feature importance의 관련성을 파악하는 metric: 
* 1) 모델에 input을 넣으면 attention weight가 나옴. 
* 2) input에 대한 prediction의 인과성을 보기 위해서 prediction을 input으로 미분함. input은 one-hot vector이고 네트워크를 통과하면서 파라미터들을 통해 gradient 값 계산할 수 있음. 
* 3) input이 바뀌면서 둘다 변할거임. 그 각 경우의 통계량 얻을 수 있음. 두 경우의 Kendall correlation 구함. 이렇게 실험했더니 상관계수 값이 0.5 근방이 나옴. 0은 관련 없는 경우이고 1은 관련 높은 경우인데, 0이 나옴. 따라서 attention weight이랑 input의 feature importance가 modest하다는 것을 알 수 있음. 

* 저자들은 이 단락을 이렇게 마무리 함.

> The results here suggest that, in general, attention weights do not strongly or consistently agree with standard feature importance scores. The exception to this is when one uses a very simple(averaging) encoder model, as one might expect. These findings should be disconcerting to one hoping to view attention weights as explanatory, given the face validity of input gradient/erasure based explanations (Ross et al., 2017; Li et al., 2016).

### 실험 2: attention이 랜덤하게 막 만들어지는 현상도 실험해볼 수 있을까? 그래서 위에 만들 모델을 변형해 봄. 
* 위의 모델이 만든 prediction과 유사하지만, attention 값이 기존과 가장 다른 방식이 되도록 다시 학습을 시킴. 결과적으로 완전히 다른 attention이 나왔지만 예측은 비슷해짐. ⇒ attention이 모델의 prediction을 설명할 수 있지 않다는 반례를 보임. 기존 attention과 adversarial attention의 분포의 차이가 Jensen shannon divergence 값이 하한이 0.69가 나옴. JSD의 범위는 0~1인 경우에 비하면 분포 차기아 꽤 난다고 할 수 이따.

* 저자들의 conclusion의 한 문장을 가지고 오면서 리뷰를 끄읕.

>These results suggest that while attention modules consistently yield improved performance on NLP tasks, their ability to provide transparency or meaningful explanations for model predictions is, at best, questionable. 

이 포스트는 이 논문의 핵심만 정리했음. 이 포스트에 언급하지 않은 내용이 많음. 예를 들어 문장의 길이가 길면 attention과 input feature importance의 관련성이 커진다, hidden state의 average을 사용하면 attention의 관련성이 커진다, 인코더가 복잡할 수록, 거대할 수록 attention의 관련성이 떨어진다, metric의 한계점, 실험의 한계점 등등...

다음 리뷰는 attention is not not explanation. 제목 부터 attention is not explanation을 저격한게 아닌가 싶음. 싸움 구경이 제일 재미있다.