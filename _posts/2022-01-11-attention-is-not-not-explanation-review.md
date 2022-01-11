---
title: "attention-is-not-not-explanation-review"
last_modified_at: 2022-01-11T16:20:02-05:00
categories:
  - nlp
  - paper-review
tag:
  - transformer
  - paper
  - NLP
---
## 직관적인 내용 정리
요약: 기존의 연구가 attention weight을 가지고 모델의 prediction을 설명하는게 위험하다, 검증되지 않았다고 주장했다면, 이 논문은... 그 논문을 저격한다. 일단 기존의 논문에서 수행한 실험의 결점들을 지적한다. 그리고 실험을 다시 해서 attention이 설명하는 부분이 있다는걸 보임. 특히 adversarial net을 기존 논문에서 완전히 잘못만들었음. 제대로 만들면 같은 결과를 가지면서 기존 attetntion과 차이가 나도록 만들 수 없다. 같은 결과를 가지도록 학습하면 attention 값도 비슷하게 됨. 즉 attention이 random 하지 않고 분명한 의미를 가지고 있다는 것. 그리고 기존의 방법들을 보완해서 attention이 실제로 모델의 prediction을 설명하는데 얼마나 도움이 되는 attention인지 측정하는 도구를 제안함. 그래서 ‘attention이 explanability에 사용할 수 있느냐’라는 질문에는 모델의 explanability에 대한 선행 연구들을 소개하면서 explanability의 정의에 따라, 모델의 task에 따라 그게 다르다... 그래서 특정 task에서는 attention이 설명성을 나타내는 상태인지 우리가 만든 도구로 검증해봐라~ 하는 논문이다. 

해본 실험

3.2

“기존 논문에서 attention에 랜덤 값이나 adversaial attention을 넣어도 성능이 비슷했다. 따라서 이 파라미터들이 큰 의미를 가지고 있지 않다”라고 한 주장을 반박하기 위해서... 아무 값이 아니라 진짜 그냥 학습하지 않는 uniform distribution 값을 그 자리에 집어넣음. 그리고 학습이 잘되는걸 보임. 그래서 기존 논문이 제기한 반례를 무력화시킴. 이 데이터는 그냥 뭘 넣어도 다 잘 된다. 그래서 이 데이터에서 advarsarial도 잘되니까 attention이 의미가 없다고 주장할수 없게 만들었음.

3.3

attention을 만들 때 random seed를 줘서 attention의 분포변화를 확인함. 이때 attention 값이 기존 모델의 값에서 분산이 커지는지 작아지는지 확인함. 분산이 작을 수록 seed가 attention에 미치는 영향이 작은거임. 그래서 attention이 실제로 특정한 방향으로 학습을 하고 있다고 볼수 있어서 어떤 의미를 가질 수 있다. 그런데 만약 seed에 이리저리 치우친다면 random일 가능성이 클 수 있음. 그래서 실험을 했더니? SST 데이터의 경우 seed에 따라 변화가 크지 않았음. Diabetes의 경우 Positive는 변화가 없고 negatiev는 컸다. 또 seed를 adversary attention에도 적용했더니, SST에서는 기존 논문의 attention 값과의 차이가 0.4 정도가 나옴. 그래서 시드를 바꿔서 adversary을 만들어도 attention 차이를 크게 만들 수 없었다!를 보임

3.4

attention이 성능 예측에 중요하다는 것을 보이기 위한 실험. 간단한 MLP 모델을 만드는데, 각 토큰의 output에 다양한 weight layer을 실험해본다. 그 weight 레이어에, 잘 학습한 attention weight도 해보고, uniform도 해보고, lstm도 해보고 이것저럿 붙여서 관찰해본다. 그랬더니 attention을 붙인 모델이 성능이 가장 좋았다. 이걸 봐도 attention이 실제 큰 역할을 하고 있다는 걸 알 수 있음. 따라서 attention parameter 안에 설명할 정보가 충분히 있다. 그리고 이 방법을 쓰면 각 task에서 현재 attention이 설명하고자 하는 정보를 가지고 있는가 확인할 수 있을 것이라고 제안함. 

1. 기존 논문이 adversary attetion을 만들 때 잘못만들었다. 그래서 그 adversary attention으로 도출한 결론이 잘못되었다. 이런 흐름을 보이고 싶어했음. 기존 논문이 잘못한 점은... data instance 마다 prediction은 비슷하게 하면서 attention 값은 다르게 학습을 시킴. 이게 instance 마다 다르게 했기 때문에 모델 자체의 attention이 아니다. 따라서 attention에 의미있는 정보가 없어요~ 라고 말할 수 없음. 그래서 이 논문에서는 모델마다 1개의 adversary attention을 학습하고자 했고, 그걸로 기존 논문의 결과를 뒤집고자 했음. 이 논문의 loss도 기존 논문의 loss와 직관적 의미는 비슷함. prediction의 차이는 줄이면서 attetion의 차이는 늘리고 싶어했음. 그렇지만 summed over instance in the minbatch으로 합쳐서 함께 loss 계산하고 학습 했음. 이렇게 했더니 비슷한 prediction을 가지면, attention 분포의 상한이 -.693이 되고 평균은 0.4 정도 되었음. 기존 논문에서 제시한, 하한 0.6? 과 반대되는 결과. 

1. explanability에 대해서... 이론적으로 설명... 하려는 말은 task에 따라 data에 따라 다양한 설명이 있을 수 있음. 설명하려는 방법도 다양하다. 다양한 해석이 있다고 해서 그 해석이 잘못되었다고 주장할 수는 없다!면서 기존 논문의 전개를 비판함...

솔직히 처음에 읽은 논문보다 잘 안읽혔음. 다른 논문을 빨리 읽어보고 싶어서 그랬는지는 모르겠지만... 무튼 이 마음을 가지면서 다 읽고 정리하느라 좀 고생했음... 이제 최신 인코더-디코더 모델로 넘어가보자! 이히히히히s