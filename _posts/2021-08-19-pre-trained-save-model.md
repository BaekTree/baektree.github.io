---
title: "0819-pre-trained-save-model"
last_modified_at: 2021-08-19T16:20:02-05:00
categories:
  - boostcamp
tags:
  - camp
---

# 트렌드
* 백본 모델을 가지고 와서 우리 데이터에 맞춰서 다시 학습하는 알고리즘이 대세. 

* 이미지는 cnn, resnet에 파인튜닝
* nlp는 transformer 기반의 bert에 파인튜닝

* 모델의 구조를 알고서 파인튜닝을 해라!

* 파인튜닝만 하면 어느 정도 괜찮은 성능을 낸다. 이미지 classification은 이제 정복했다고 여겨짐

* 학습 결과를 공유하고 싶을 때. 모델을 동료한테 교수님한테 다른 연구자한테 주고 싶을 때. 

* 저장 먼저 해야 한다. 토치에서는 뭘 제공할까?
  * model.save() 함수

* 2가지 방법

* 모델 형태 아키텍처를 저장

* 파라미터를 저장. 위 보다는 용량이 적다. 이거만 넣으면 다시 돌릴 수 있다. 

* 중간 과정의 저장을 통해 최선의 결과를 선택도 가능(early stoping or hyper parameter choice)

* save 자체는 쉽다. state_dict: 파라미터 상태 표시.

* 이것을 torch.save에 arg으로 넣어서 저장.

* 파이토치에서 모델 저장할 떄는 pt확장자. 파이썬의 기본 확장자. 

* state_dict을 ㅓ장했을 때 불러오기 load_state_dict.으로 불러옴. 

* 아키텍처랑 함께 ㅓ장(torch load model path). 잘 안씀. 파이썬 pickle 방식으로 저장. 주로 파라미터만 저장.

* checkpoints
* 중간 결과 도중에 계속 저장
* ealry stoping 때 사용

* epoch, loss, metric함꼐 저장


* transfer learning
* 다른 데이터로 만든 모델을 현재 데이터에 적용
* 대용량일수록 성능이 좋다. 그래서 백본 모델 사용.
* 이미지넷
* 버트, gpt
* 이미 만들었다고 pre training 모델
* 남이 잘 만든것을 내 데이터에 추가 학습

* torchvision은 cv 모델
* hugging face는 nlp 표준

* freezing: 특정 부분은 파라미터를 학습시키지 않는다. 
* 요즘은... k fold 처럼 레이어 하나 하나... freeze 풀고 학습... 반복...


* 모니토링 도구
* 텐서플로우에서 텐서보드 제공하면서 학습 중에 acc와 loss 트래킹 가능.

* 학습하면 시간이 길다. 학습 시키고 자고 일어남. 시간 동안 기록이 필요. print와 log을 쓰거나 csv을 기록 함. 텐서보드와 weight&biases을 사용함. 텐서보드는 유명. 

* 간단한 사용법
* 텐서플로우의 시각화도구
* 학습 그래프, metric, 학습 결과 시각화
* 토치도 연결 가능

* 텐서보드 값
* scalar: metric, epoch... precision, recall... 
* graph: 보기 좋게 표시
* histogram: weight의 분포를 표현. 적절히 값이 분포 중인가
* image: 예측 값과 실제 값 비교 표시

* 코드
* log 디렉토리 설정
* torch summaryWriter import 기록 객체

* witer instance
* 씀 각 지표들
* loss 카테고리 안에 train 값/ test 값. 
* slash가 카테고리이다


* 과제 custom model
* torch.gather

* 직관: 1개를 가져올 때 1개의 index가 필요하다. 가져올 개수 만큼의 index 개수가 있어야 함. 

* dim = ?을 기준으로 1개 씩 가져온다고 생각해야 한다. 

* input차원가 indices 의 차원이 같아야 한다. 
* dimension을 맞춰줘야 한다. 
* (2,2,2)가 입력이라면... ()

* 3d

* 예제
주어진 것. RNN에서... 8개의 batch. 각 sequence 길이. 각 hidden state.

뽑고자 하는 것

3차원
```
import torch

A = torch.Tensor([[[1, 2],
                   [3, 4]],
                  [[5, 6],
                   [7, 8]]])

# torch.gather 함수를 써서 해보세요!
output = torch.gather(A,2,torch.tensor([[[0],[1]],[[0],[1]]])).view((2,2))
# print(output)
"""
A: 2,2,2
[[[0],[1]],[[0],[1]]]
target sahpe: 2,2
0
1
0
1
"""

# 아래 코드는 수정하실 필요가 없습니다!
if torch.all(output == torch.Tensor([[1, 4], [5, 8]])):
    print("🎉🎉🎉 성공!!! 🎉🎉🎉")
else:
    print("🦆 다시 도전해봐요!")
```

임의의 3차원

원하는 결과
```
A = torch.tensor([[[ 1,  2,  3],
                   [ 4,  5,  6],
                   [ 7,  8,  9],
                   [10, 11, 12],
                   [13, 14, 15]],
        
                  [[16, 17, 18],
                   [19, 20, 21],
                   [22, 23, 24],
                   [25, 26, 27],
                   [28, 29, 30]],
        
                  [[31, 32, 33],
                   [34, 35, 36],
                   [37, 38, 39],
                   [40, 41, 42],
                   [43, 44, 45]]])

if torch.all(get_diag_element_3D(A) == torch.Tensor([[ 1,  5,  9],
                                                     [16, 20, 24],
                                                     [31, 35, 39]])):
```

dim = 1에서부터 정방행렬의 크기까지만 반환.

indices을 어떻게 구할까?

a,b,c = 

[[[0]] [[1]]]



Q 과제 param.data으로 변경할 때 data keyword가 어디에서 나옴?

https://pytorch.org/docs/stable/_modules/torch/nn/parameter.html#Parameter

parameter instance을 생성할 떄 data property에 attribute으로 저장.