---
title: "2021-04-18-Metrics-Review"
date: 2021-04-18T15:34:30-04:00
categories:
  - machine learning
tags:
  - machine learning
  - metric
---

# metric 정리

* 갖가지 metric을 정리해본다.

* 동기: ML 프로젝트를 하다가 issue가 발생
* 새로운 data가 등장해서 모델이 negavite을 Flase Positive으로 인식한다.
	* 해결책 : metric을 수정. 일단 과녁을 바꿔야 한다.
	* metric을 바꾸고 나서는 해당 문제를 해결하기 위해 새로 학습 시켜야 한다. 비용 함수도 수정해야 한다.


# Wikepedia
## precision and recall and other metrics
* relevent : 유저가 필요로 하고 목적이 되는 정보. 맞는 것. 옳은 것.
* retrieved : 프로그램, 기계, 머신이 쿼리의 결과로 내놓은 것. : true라고 기계가 판단한 것.
* precision = positive predictive value
  * fraction of relevant instances among the retrieved instances. 
  * 맞다고 예측한 것들 중에서 맞은 것의 비율
  * 그래서 정확도라고 부름
  * y_pred 중에서 y = True의 비율
  * 실제 y = True에서 놓친 것은 파악할 수 없음.

$$
	Pricision = \frac{TP}{TP + FN}
$$
  
* recall = sensitivity = hit rate
  * 맞는 것들 중에서 맞다고 예측한 비율
  * fraction of relevant instances that were retrieved.
  * releveant instances 의 비율. retrived 한. 전체 relevant instance에서 retrived된 것. 맞는데 맞다고 예측된 것.
  * y = True 중에서 y_pred = True.
  * Y = False인데 y_pred을 True라고 틀리게 예측한 것은 파악할 수 없음.

$$
	recall = \frac{TP}{TP + FN}
$$

* 검색 엔진 예시: 
  * 사용자가 날린 쿼리에 relevent한 전체 데이터가 DB에 60개가 있다. 엔진은 20개의 relevent한 데이터와 10개의 irrelvent한 데이터를 같이 사용자에게 리턴했다.
  * precision : positive predictive value = true positive/retrived = 20/30 = 2/3
  * recall : sensitivity = 20/60 = 1/3

* accuracy
  * data set이 imbalance하면 accuracy을 신뢰할 수 없음.
  * True : False의 비율이 9:1이면 그냥 count을 해도 90% accuracy가 나옴.
  * 다른 metric을 사용해야 함.
  
$$
	\frac{TP + PN}{TP + FP + PN + FN} = \frac{correct True and False}{Total}
$$

* f1
  * recall과 precision을 harmonic mean한 것
  * 예측한 y_pred의 정확도 precision과 y = True에서의 실제 맞춘 것을 적절히 파악함.
  * * $0 \le f_1 \le 1$

$$
	2 \cdot \frac{precision \cdot recall}{precision + recall}
$$

* $f_{\beta}$
  * beta = 2
    * recall의 가중치를 precision보다 더 크게 준다.
  * beta = 0.5
    * recall의 가중치를 precision보다 더 작게 준다.

$$
	(1 + \beta^2) \cdot \frac{precision \cdot recall}{(\beta^2 \cdot precision) + recall}
$$

* ROC
  * binary classifier은 결국 확률을 True 혹은 False으로 구분해야 한다. 그 기준 threshold을 보통 0.5
  * 0.5보다 크면 True, 작으면 False
  * 보수적으로 하려면 True라고 잡는 기준이 더 높아야 한다. 
    * 보수적으로 잡을 때는 0.5보다 더 크게 잡는다.
  * threshold을 변화할 때 성능을 개괄적으로 나타낸 그래프가 ROC curve
  * 그래프을 구성하는 각 점이 하나의 threshold일 때 모델의 성능을 나타낸다.
  * $0 \le threshold \le 1$에서 성능을 구하면 그래프가 됨~
  * x 축은 FPR
    * $\frac{FP}  {FP + TN}$
    * 실제 negative인데 posotive라고 잘못 예측해서 가지고 간 것이다.
    * recall에서 잡지 못한 것. recall에 잡히지 않는 값이다.
  * y 축은 TPR
    * TPR = racall
    * 전체 Y = True 중에서 제대로 예측 한 것.
  * 45도 선은 학습 없이 1/2의 활률로 구분하는 것.
  * 45도 선보다 위에 있으면 1/2 보다 잘 예측하는 것
  * 45도 선보다 아래에 있으면 1/2 보다 못 예측하는 것.
  * TPR = 1이고 FPR = 0 인 좌상향 지점의 점은 True Positive을 다 맞힌 것이고 negative 중에서도 positive으로 잘못 예측한 것이 없는 경우이다. 모두 맞힌 것.
  * threshold = 0이면, classifier가 어떻게 확률을 예측해도 모두 True라고 결정내린다. 
    * y = True 값에서 y_pred 값을 모두 True라고 예측할 것이다.
      * 따라서 TPR = 1
    * y = False 값도 모두 y_pred에서 True로 예측
      * FPR = 1
    * 따라서 $ROC = \frac{TPR}{FPR} = 1$이 되어 그래프의 우상향 끝에 위치.
  * $0 \lt threshold \lt 1$	
    * TPR을 1에 가까울 수록 좋다
    * FPR은 0에 가까울 수록 좋다
    * 따라서 y축은 커질수록 좋고
    * x축은 작아질 수록 좋다
  * thereshold = 1이면 모든 값을 False으로 예측한다.
    * y = True 값에서 y_pred 값을 모두 False라고 예측할 것이다.
      * 따라서 TPR = 0
    * y = False 값도 모두 y_pred에서 False로 예측
      * FPR = 0
    * 따라서 $ROC = \frac{TPR}{FPR} = 1$이 되어 그래프의 좌하향 끝에 위치.

![ROC from wiki](/assets/src/metric/ROC.png)

  * https://en.wikipedia.org/wiki/Receiver_operating_characteristic
  * https://www.youtube.com/watch?v=4jRBRDbJemM
* AUG
  * binary classifier을 만들 때 다양한 모델을 사용할 것.
  * logistic regression
  * random forest
  * 등등
  * 각 모델에 대해서, $0 \le threshold \le 1$에서 그래프를 그릴 수 있음.
  * 각 모델의 성능 비교를 한눈에 할 수 있다.

더 많음. 
참조
https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers




## statistics review
* null hypothesis : given item is irrelevant. H0. no affect. no difference. Two are same. mu1 = mu2. two distributions are same. 지켜야 할 가설. 새로운 시도, 알고자 하는 것이 아닌 것. 기존의 것. negative 상태. 보수적인 것. 지키는 것. 방어하려는 의견.
* alternative hypothesis : 새로운 것.  알고자 하는 것. 판단하고자 하는 것. 공격하려는 의견. 
* positive : 공격에 성공했다. negative 공격에 실패했다.
* false positive : 새로운 대안이 옳다고 결정했음. null을 reject했음. 그러나 사실은 null이 맞았음. 공격에 성공했음. 그러나 사실은 실패가 맞음.
* false negative : null의 손을 들어줬음. 기존을 유지하고자 함. 그러나 사실은 대안이 맞았음. 공격에 실패. 하지만 사실 성공이 맞음. 
* null hypo는 기존의 의견. 기존의 입장. alternative hypo가 대결을 신청 함. 우리는 alternative가 옳은지 틀린지 판단할거임. 옳다고 판단하면 positive, 틀리다고 판단하면 netagive. hypo에 대한 rejction을 판단. positive -> reject null. negative -> no reject null.
* type 1 error : 대안의견이 옳다고 판단했는데 사실은 null이 옳았던 경우. false positive.
* type 2 error : 대안의견이 틀렸다. null의 기존의견이 옳다고 결정했지만 사실은 대안의견이 옳았던 경우.

* null hypo : given item is irrelevant.
* irrelavant인데 가져갔다면 type 1 error
* relavant instance인데 안가져 갔다면... type 2 error.

* recall : relevant instances 중에서 몇개를 가지고 갔냐?