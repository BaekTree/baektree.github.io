---
title: "marginal rank loss"
last_modified_at: 2022-01-12W16:20:02-05:00
categories:
  - nlp
tag:
  - NLP
  - rank
  - loss
---

# marginal rank loss

siamness, triple loss, similarity loss, rank loss 모두 다 같은 개념. 다른 task에 쓰여서 다르게 불린다. 

## facial recognition example

CNN facial recognition task의 경우... anchor positive, anchor negative 이렇게 데이터가 들어온다. anchor positive는 두 벡터의 차이가 적어야 하고 anchor negative는 두 벡터의 차이가 커야 함. 

다시 말해서 anchor vector와 positive vector은 유사해야 한다 그리고 anchor vector와 negative vector은 유사하지 않아야 한다. 

유사성을 구하는 함수를 d, anchor vector을 a, positive vector을 p, negative vector을 n이라고 하자. 그러면 모델이 예측을 잘 할수록 $d(a, p)  - d(a, n)$의 값은 negative가 된다. $d(a, p)$의 값은 작을 것이고 $d(a, n)$의 값은 크니까. 반대로 예측을 잘못하게 되면 $d(a, p)$의 값은 크고 $d(a, n)$의 값은 작다. 따라서 $d(a, p)  - d(a, n)$의 값이 positive가 됨.

학습을 하는 관점에서 보면, 예측을 잘할 수록 loss의 크기는 작아야 하고, 반대로 예측을 잘못할수록 loss의 크기는 커야 한다. 그 값을 바탕으로 gradient descent을 구함.

따라서 loss을 이렇게 정의할 수 있음. 

$$
L(a, p, n) = max( 0, d(a, p) - d(a, n) )
$$

그런데 이 상태의 loss는 한가지 예외 사항이 있음. a와 p 차이 보다 a와 n의 차이가 상대적으로 가깝긴 한데... 그래서 $d(a, p) - d(a, n)$ 의 값이 negative가 되기는 했지만... 만약 그 차이가 사실 얼마 안나는 것으로 모델이 예측을 했다면? 사실은 그 차이가 많이 나는데 말이다... 그러면 실제로 ground truth에는 큰  차이가 있지만 loss는 0이 되어서 학습을 안하게 됨. 

이런 문제를 극복하게 위해서 한가지 파라미터 m을 추가 함. 

$$
L(a, p, n) = max( 0, d(a, p) - d(a, n) + m )
$$

저 두 차이가 m 보다는 커야만 진짜 제대로 맞춘거라고 인정 해준다. 잘 맞춘 차이가 m보다 훨씬 클 때, m을 더해도 음수가 되면 loss는 0이 된다. 그런데 그 차이가 m보다는 작으면... m을 더했을 때 양수가 되어서 loss가 양수가 되고, 그만큼을 모델은 더 학습하게 됨. 

만약 완전히 잘못예측해서 $d(a, p)$가 $d(a, n)$ 보다 작게 되면? 그럼 그 상태로도 loss가 positive인데, 거기에서 m만큼의 크기를 추가로 더 잘못하게 됨. 

## relative rank example
* 배경: 
악성 댓글 같은 경우 binary classificatoin을 넘어서 실수 범위로 각 toxicity을 예측하는 모델 / 데이터ㅔㅅ을 만드는 건 어렵다. 명확한 기준을 만들기 어렵고 annotator 마다 그 정도도 다르게 느끼기 때문이다. 그래서 각 악성 댓글 sample을 두개씩 비교 함. 상대적으로 어떤 댓글이 더 toxic한지를 annotator가 판단하게 한다. N개의 샘플이 있으면 n^2개의 총 경우의 수가 나오게 됨. 이 작업을 하고 나서 각 샘플 마다 n-1개의 상대적인 toxic 결과가 나옴. 이 결과를 바탕으로 전체 sample에서 ranking을 매길 수 있음. n*(n-1)의 복잡도을 가진다. 너무 느려서 개선 방안도 제시가 되었음. BWS는 1991년에 처음 나왔다. annotator들한테 n tuple의 데이터를 준다. 보통 n = 4이고 n은 무조건 1보다는 크다. 그러면 annotator는 n개 중에서 조건에 맞는 best 1개와 worst 1개를 뽑는다. 그리고 best와 worst로 다른 n-2개의 원소들과 비교 표현으로 나타냄. A가 best이고 B개 worst이면, A > C, A > D, A > B, C > B, D > B. 그러면 항상 5개의 선호 표현으로 나타낼 수 있음! 이걸 real value로 전체 데이터에 대해서 나타내는건... items 사이와 property of interest(연구 주제. 지금 우리는 악플 정도)에 따라 계산될 수 있다... Kir-itchenko and Mohammad (2016)에서 제안한 거라고 함.

* 모델
모델이 학습하는 것은 두개의 악성 댓글 pair가 주어질 때 어떤 댓글이 어 심한지를 학습해야 함. 그래서 위의 
$L(a, p, n) = max( 0, d(a, p) - d(a, n) + m )$에서 유사도 함수 $d$ 대신에 각 댓글의 악성 정도의 실수 값이 들어가면 됨. 앞 항이 덜 toxic하고 뒤가 더 toxic 하면 차이가 negative가 되어서 잘 학습하고 있는 상태가 된다.


## 그래서 pytorch는?

파이토치는 margin rank loss 모듈을 제공함. 이렇게 생김. 

$$
loss(x1,x2,y)=max(0,−y∗(x1−x2)+margin)
$$

이렇게 하면 위에 적은 최종 공식이랑 조금 다름. $-y$ term이 추가 됨. 그런데 이 부분은 그냥 x1와 x2 중에 어느 것이 positive이고 negative인지 나타낼 때 쓰는 거임. 
위에 적은 $L(a, p, n) = max( 0, d(a, p) - d(a, n) + m )$의 예시를 보면, $d(a, p)$이랑 $d(a, n)$ 중에 ground truth 값은 앞항이 작고, 뒤 항이 커야 함. 그래서 $d(a, p) - d(a, n)$값이 negative가 나와야 잘 나오고 있는 상태라고 했음. pytorch의 수식에 적용하면, x1이 작아야 하는 갑싱고 x2가 커야 하는 값이면 x1-x2가 negative가 되어야 함. 따라서 y의 값이 -1이 되어야 -y가 positive가 되고 x1-x2가 negative으로 유지된다.만약 반대로 x1이 커야 하고 x2가 작아야 하는 값이면 y값을 1을 두면 됨. 

[https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html](https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html)




끗.