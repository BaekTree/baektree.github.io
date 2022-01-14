---
title: "Contrastive Learning for Many-to-many Multilingual Neural Machine
Translation(ACL 2021) review"
last_modified_at: 2022-01-19M16:20:02-05:00
categories:
  - nlp
  - paper-review
tag:
  - paper
  - NLP
---
# Contrastive Learning(ACL 2021) review

Contrastive Learning for Many-to-many Multilingual Neural Machine
Translation 핵심 요약

## 제출한 사람들:

bytedance AI에서 ACL 2021에 제출한 논문.

## 배경:

NMT에서 여태 성능이 좋은 모델들은 특정 한 언어에서 다른 특정 한 언어로 번역을 학습시키는 bilingual model들이었음. mBERT 이후로 XLM, mBART등 학습할 때 한꺼번에 다양한 언어를 학습시키는 multligual language model이 나오고 있긴 한데, 항상 데이터가 많은 두 언어를 학습한 bilingual점수에서 기존의 bilingual만 학습한 모델에 비해서 성능이 밀렸다. 

## 저자들이 이뤄낸 일(contribution):

multilingual 언어를 학습시켜서 다양한 번역을 할줄 아는 모델을 만들면서, biligual 언어 번역에도 거의 뒤쳐지지 않는 성능을 냈다! 이제 하나의 모델로 많은 언어를 모두 번역할 수 있는 희망의 길이 보인다!

## 저자들이 세운 가설:

언어가 다양해도 사실 의미는 같을 것이다. 따라서 언어 별 단어나 맥락이 다르다고 해도 동일한 feature representatoin을 모델이 알고 있으면 여러 언어들을 모두 잘 번역할 수 있을 것이다.

## 학습한 데이터:

PC32(32 English centric parellal corpora: multilingual dataset), MC23(저자들이 직접 만듦)

## 저자들이 한 실험:

1. contrasive learning

동일한 feature representation을 학습 시킬 때 필요한건 뭐다? 그래서 contrasive learning 적용했음. 따라서 필요한 데이터는  A언어의 anchor text가 있으면, 다른 B 언어의 positive tex와 B언어의 random한 negative text을 준비해서 한 세트의 데이터를 만든다. NMT의 인코더에 임베딩 할때 output이 두개로 만든다. 하나는 기존의 트랜스포머 처럼 디코더도 QKV을 보내는것 말고 새로운 브랜치에서 이 contrasive learning의 loss을 뱉는다. 이 loss랑 디코더에서 나오는 loss랑 결합해서 최종 loss가 됨. contrasive learning loss는 anchor과 postive, negative간 유사도를 구해서 softmax 태운다. positive sample과 anchor에 해당하는 데이터는 정답 레이블 true에, negative와 anchor의 유사도는 false에 가도록 cross entropy으로 학습. 여기에 파라미터 tau가 있음. margin rank loss에서 margin을 담당하는 역할 같다. 

![SmartSelect_20220119-204254_OneDrive.jpg](/assets/src/Contrastive Learning(ACL 2021)/SmartSelect_20220119-204254_OneDrive.jpg)

1. augmentation

Aligned Augmentation을 해서 데이터를 불렸다. 근데 이 방법이 그냥 데이터를 불려서 generalization을 크게 하는 것 말고 더 큰 특징이 있음. 이게 multiligual language을 한꺼번에 학습해야 하기 때문에 거기에도 도움을 주도록 함. A 언어의 text 데이터에 각 단어 토큰들이 나열 되어 있으면, 랜덤하게 특정 토큰을 다른 언어로 바꿈. 그래서 같은 의미의 자연어 단어가 같은 의미의 hidden representation을 가지도록 도와줌. 

![SmartSelect_20220119-204346_OneDrive.jpg](/assets/src/Contrastive Learning(ACL 2021)/SmartSelect_20220119-204346_OneDrive.jpg)

1. 학습 환경

gradient norm 쓰고 V100 8 * 4개인가? 쓰고... hidden dim 1024늘리고 layer 12개로 늘리고 warm up 10000정도 쓰고 등등...

## 성능

![SmartSelect_20220119-204656_OneDrive.jpg](/assets/src/Contrastive Learning(ACL 2021)/SmartSelect_20220119-204656_OneDrive.jpg)

주어진 evaluation dataset에 대해 supervised learning을 한 결과 테이블이다. 맨 아래의 mRASP2가 이 저자들이 만든 논문. 맨 위 bilingual transforer 12의 En-Fr을 일단 넘었음. multilingual langauge을 통채로 학습하면서 동시에 English centric한 성능을 넘었음. 그러면서 동시에 다른 언어들에서 대부분 SOTA을 달성 함. 게다가 기존의 multilingual transformer 모델들(중간의 pretrain & fine-tune row에 해당하는 모델들)을 대부분 뛰어넘거나 비슷비슷한 점수에 도달.

![SmartSelect_20220119-204711_OneDrive.jpg](/assets/src/Contrastive Learning(ACL 2021)/SmartSelect_20220119-204711_OneDrive.jpg)

Unsupervised learning의 결과. 주어진 evaluation dataset에 대해서 pretrained 만으로 성능이 다른 모델들에 비해 조음.

![SmartSelect_20220119-204721_OneDrive.jpg](/assets/src/Contrastive Learning(ACL 2021)/SmartSelect_20220119-204721_OneDrive.jpg)

영어가 아닌 번역에 대한 zero-shot 결과 table이다. 현재까지 가장 좋은 성능을 냈던 것은, A언어를 영어로 바꾸고 다시 영어를 B언어로 바꾸는 pivot 방법이 성능이 좋았는데... zero shot으로 pivot 없이 pivot에 근접한 성능을 만들어냄.

## 앞으로 저자들이 하고 싶은 것

지금은 32개 언어로 핛브했는데 100개 이상의 언어로 늘리고 싶다고 함.

느낀점: contrasive learning으로 같은 의미의 토큰을 학습시킨다는게 좀 신기. 그리고 동일한 의미의 토큰을 다른 언어로 바꿔 augmentatoin 한것도 창의적인것 같았음. 그리고 제안한 방법이 성능이 좋다는 걸 꽤 엄격하게 검증하려고 한 것 같았음. 다른 상황에서 동일한 방법론을 적용할 방법을 찾는게 쉽지 않았을 텐데... supervised, unsupervised, zero-shot 등등에 동일한 방법을 다 적용해 봄. 

끗.