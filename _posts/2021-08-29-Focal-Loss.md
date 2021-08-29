---
title: "08-29-Focal-Loss"
last_modified_at: 2021-08-06T16:20:02-05:00
categories:
  - Blog
  - deep learning
  - loss function
tags:
  - focal loss
---

# focal loss

데이터에 imbalance가 있을 때 해결하기 위해 사용하는 loss...? 인줄 알았는데 사실은...

# paper review focal loss

Facebook ai research팀에서 오잉!하고 놀랐고 그 카이밍 헤가 공저자로 참여해서 ㅗㅜㅑ하고 놀랐음.

## 요약: 
cnn을 사용할 떄, one stage object detection에서도 focal loss와 RetinaNet을 쓰면 높은 성능을 낼 수 있다.

## 배경: 
one stage decoder(YOLO 등)은 속도는 아주 빠르다. 하지만 상대적으로 정확도는 떨어짐. two stage decoder(R-cnn 계열)은 속도은 조금 떨어지지만 정확도는 상대적으로 더 높았다.

## 좀더 상세 요약: 
이런 일이 발생하는 이유는 이미지에서 object 보다 배경이 훨씬 많다는 특성에서 비롯 됨. 이미지의 대부분의 정보는 background. 실제 object가 있는 foreground는 얼마 되지 않는다. 즉 데이터 불균형이 항상 존재하는 것. 그래서 cross entrpy loss을 구하면 background에서 발생한 loss가 훨씬 많아서 forground loss의 영향이 미미해져서 학습이 잘 되지 않는 것이다. two stage decoder의 first stage 먼저 전체 이미지를 나눠서 실제 object가 있는 patch을 가려낸다. 두번째 stage에서 각 후보 패치들만 살펴보면서 실제 object가 있는지 판단함. 이것을 one stage만에 해결하기 위해 소개하는 것이 focal loss이다.

아이디어: 학습이 잘 되는 것의 가중치를 낮추고 학습이 안되는 것의 가중치를 높인다. 기존의 literature에서는 반대로 했었음. 쓸모없는 background 데이터의 가중치를 낮춰서 유용한 데이터를 학습(outlier의 가중치를 낮춤). 그런데 focal loss에서는 반대로 학습이 잘 되는 가중치를 낮추고 안되는 가중치를 높임. 이 아이디어를 기존의 cross entropy에 살짝만 수정해서 적용 함.

수식은 이케 됨. 

$$\operatorname{FL}\left(p_{\mathrm{t}}\right)=-\left(1-p_{\mathrm{t}}\right)^{\gamma} \log \left(p_{\mathrm{t}}\right)$$

현재 실제 레이블에 대하여, softmax가 예측한 이 레이블의 확률이 $p_t$이다(다른 틀린 레이블의 값들은 0이 됨!).$\gamma$는 hyper parameter이다. 논문에서는 2으로 줬더니 CE 보다 더 좋은 성능을 냈다고 함. 예측을 잘할수록 $p_t$가 1에 가까우면 gamma 항이 0에 가까워져서 더 작아짐. 그래서 loss 항 전체가 작아짐. 반대로 예측을 잘 못할수록 $p_t$가 0에 가깝고 gamma 항이 유지 됨. 따라서 loss가 그대로 유지된다.

여기에 새로운 hyper parameter $\alpha$까지 추가해서 성능을 더 끌어올렸다고 함(왜 추가하게 되었는지는 안알려줌. 그냥 실험했더니 더 좋더라...). 

$$\mathrm{FL}\left(p_{\mathrm{t}}\right)=-\alpha_{\mathrm{t}}\left(1-p_{\mathrm{t}}\right)^{\gamma} \log \left(p_{\mathrm{t}}\right)$$

그리고 여기에 sigmoid까지 넣어서 0과 1사이의 범위 안에 오도록 안정성까지 추가.

이후에는 Retina Net에 대한 설명과 구현까지 나오는데, 당장은 NLP 쪽에 더 큰 관심이 있어서 다음을 위해 미뤄둠.

실제 저자들은 focal loss을 Retina Net에 적용해서 one stage decoder임에도 다른 two stage decode의 성능을 뛰어넘는다는 실험 결과를 보임. 

CE의 Loss와 focal loss의 CDF을 비교해보면 충분한 데이터의 loss는 gamma의 영향을 잘 받지 않지만 데이터가 없는 레이블의 loss는 gamma에 따라 영향을 많이 받아서 imbalance을 잘 해결해 준다.

굉징하 친절한 논문이었음. 미분까지 다 계산해서 알려줌. 배경도 잘 알려줬음. 

https://arxiv.org/pdf/1708.02002.pdf