---
title: "Maximum-weighted-liklihood-estimation"
date: 2021-09-01T15:34:30-04:00
categories:
  - machine-learning
tags:
  - machine-learning
  - loss
  - data imbalance
  - Maximum-weighted-liklihood-estimation
  - Statistics
---

# Maximum-weighted-liklihood-estimation review

## 동기
  * 상황에 따라 DL에서 cost function에 가중치를 부여함
  * imbalance dataset에서 자주 쓰던데...
  * keras의 `class_weight` 사용
  * 직관적으로는 data set이 부족한 class에 비용을 크게 줘서 파라미터를 많이 이동시킴. 오케이... ㅇㅈ
  * 그런데 세부적인 의미에서는 어떻게 동작하는지 호기심이 생겨서 이것 저것 찾아보다가 이 논문을 발견 했음
  * 잼

## 내용
* 어떤 population의 확률분포 파라미터를 학습하고 싶음.
* 그런데 샘플이 몇개 없음...
* MLE의 특성 상 데이터의 수가 적으면 그 만큼 잘못된 추정을 하게 됨.
* 단적인 예로 1/2 코인을 백만번 던져서 카운트 하면 아 확률이 대강 1/2 되는구나 알수 있음. 그런데 딱 다섯번을 던졌는데 뒷면 1번 앞면 4번 나옴. 그러면 베르누이 분포 파라미터 추정 0.2로 하는 거임.
* Law of Large Number? Central Limit Theorem? 샘플의 수가 너무 적음 ㅜㅜ
* 수학적으로는... 잘못된 파라미터로 추정할 확률이 높아짐.
* 이 문제를 해결하기 위한 Idea : 분포가 비슷해보이는 다른 population들에게서도 샘플을 뽑고 정보를 취합하자. 알고 싶은 population의 정보의 부족한 부분을 다른 비슷한 population에서 얻자!
* 그래서 다른 population들에게서도 샘플을 얻고, 객 population에 다른 가중치를 부여 함.
* 모든 샘플들을 가중치로 합쳐서 Liklihood Estimation을 한다!


$$
    \mathcal{L} = \prod^m \mathcal{L}(y_1, \hat y_1;\theta)^{\lambda_1}\mathcal{L}(y_2, \hat y_2;\theta)^{\lambda_2}\\
    \mathcal{l} = \sum^m \lambda_1 \log \mathcal{L}(y_1, \hat y_1;\theta) + \lambda_2 \log \mathcal{L}(y_2, \hat y_2;\theta)\\
    \argmax_{\theta} \mathcal{L}\\
    \frac{\partial \mathcal{L}}{\partial  \theta} = 0\\
$$

* optimul value을 찾을 때 가중치에 따라 보정되어 최적 파라미터를 찾아낸다.
* 논문에서 엄밀하게 전개해 나갈 때는 population이 normal distribution 혹은 bivariate normal distribution일 경우에, Loss Function을 MSE으로 가정하고서, 비슷한 population의 조건들이 어떤 것인지, 그리고 어떤 가중치를 부여할 때 optimal point에 도달할 수 있는지 등등의 조건들을 엄밀하게 보임...
* 무려 2001년 논문... ㄷㄷㄷㄷ



## ref

https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiqv4bU5LLwAhWNzpQKHaY1AIkQFjAAegQIBRAD&url=https%3A%2F%2Fopen.library.ubc.ca%2Fmedia%2Fdownload%2Fpdf%2F831%2F1.0090880%2F1&usg=AOvVaw1e-vLYKmYTeKjnzmWDFmcn