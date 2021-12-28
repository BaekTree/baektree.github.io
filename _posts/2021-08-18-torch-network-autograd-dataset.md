---
title: "0818-torch-network-autograd-dataset"
last_modified_at: 2021-08-18T16:20:02-05:00
categories:
  - boostcamp
tags:
  - camp
---
과제 1 질문
backward hook, forward hook에 input output 모두 있다. 굳이? forward pre hook에는 input만 있음. 

* Q: 왜 hook에는 둘 다 있을까 어떤 쓰임새가 있을까? resudial network? 



* 토치 텐서는 map이 안된다!
```
# below is impossible
out = map(lambda x:x / torch.tensor([sum]),grad_input )
```
안됨. 토치는 맵이 안된다고 함. 대안이 있을까?

# 토치로 네트워크 만들기
* 실제로 쌓아보자
* inception net 아키텍처를 보면 사실 반복의 연속.
* 레이어들을 하나 하나 쌓아서 다음으로 넘긴다.
* 하나의 블록을 정의해서 정보를 다음으로 넘기고 백 pp할 때 다시 전달.

## e.g. transformer
* layer: softmax, linear, ff, normalization, multihead attension
* 각 레이어를 합쳐서 블럭을 만든다. 인코더 레이어, 디코더 레이어

## e.g. resnet
* layer: norm, conv
* block: layer + identity

## 블록 만들기: 
* 토치의 가장 기본적인 class는 nn.Module
* 4가지를 정의해야 한다
  * input 
  * output 
  * forward 
  * backward(augo grad 자동 미분 weight. 학습 대상)
+ 학습 대상 파라미터 정의
* weight은 파라미터 객체에 정의.
* 파라미터는 텐서의 상속 객체
* attribute가 될 때 requred_grad = True으로 해야 학습 대상이 된다.
* 직접 지정할 일을 잘 없다. 그냥 레이어에서 init하고 잘 씀.

```
model.parameters()#: summary without name
model.named_parameters(): summary#: with name

# with param name, get the instance. get names with named_parameters
model.get_parameters('nameParam')
```

* backward 함수
  * y와 hat y차이에 대해 미분하는 함수
  * 해당 값으로 파라미터 업데이트

* autograd할때 이 backward가 호출되는 것

* zero_grad 이전 값 초기화. 이전 grad가 현재에 영향 미치지 않도록

* 5가지 step
  * grad_zero
  * forward
  * loss
  * backward: 미분계산해서 새 미분값을 assign
  * optimize.step: update


## how loss, optim are related?
* loss.backward
* optim.step
* A reference to the backward propagation function is stored in grad_fn property of a tensor.

* requires_grad_(True)을 하면 tensor의 grad_fn property에 grad 수행 함수를 reference으로 넣는다.

* To compute those derivatives, we call loss.backward(), and then retrieve the values from w.grad and b.grad. 
* loss.backward을 부르면 각 tensor의 grad property에 grad값을 저장한다. 

* 관찰: loss도 tensor을 반환한다. 그리고 grad_fn pp가 꼭 있음. 아마 tensor에 backward 함수가 있는 것이고. 그래서 이것을 실행하면... grad_fn에서 계산함. 그리고 auto grad의 backward을 실행시키고... ctx을 따라서 이전의 텐서들로 이동하고 다시 pp의 grad fn 계산 반복...

* loss.backward 이전에 optimer(model.parameter)을 하면 파라미터들의 레퍼런스를 optim에 저장해둔다. 각 파라미터들은 텐서들이고 grad_required = true이라서 pp에 grad와 grad_fn을 가지고 있음.

* optim.zero_grad하면 각 파라미터의 텐서의 grad pp을 초기화 함.

* backward하면 파라미터의 텐서의 grad pp에 assign.

* optim.step 하면 grad pp의 값을 parameter의 텐서 값으로 assign.



```

model = Model(...)

repeat iteration for each epoch

    y_hat = model(...data...)

    optim = optimizer(model.parameter)

    loss = loss_func(y_hat, label)

    print(loss)

    loss.backward()

    optim.step

```




## Further Question
1 epoch에서 이뤄지는 모델 학습 과정을 정리해보고 성능을 올리기 위해서 어떤 부분을 먼저 고려하면 좋을지 같이 논의해보세요
optimizer.zero_grad()를 안하면 어떤 일이 일어날지 그리고 매 batch step마다 항상 필요한지 같이 논의해보세요

# 토치 데이터셋

* data centric ai: 어떻게 대용량 데이터를 잘 넣을 것인가?

* Q: data centric ai with generative model?

* 파티토치 데이터셋 api: 쉽게 데이터를 모델에 feeding 해줄 수 있다.

* 순서: 저장-> dataset -> transform -> dataloader
* 데티어를 모아서 전처리해 둔 후 어디에 모아둠. 

* 데이터셋 class: 
  * 시작. 
  * 크기. 
  * get item()데이터 받을 떄 어떤 형식으로 반환? index 어떻게 등등... 전처리를 여기서 하지는 않는다!

* transform data: 자료구조를 텐서로 바꿈. 추가적인 전처리도 transform에서 하기. 

* 1개 1개의 형식을 정의해서 dataset에서 나오면 dataloader으로 배치로 묶어준다. 미니배치 설정도 등등.

* dataset class
  * 입력형태 정의 init. label=... data = ... + 데이터 디렉토리 정의.
  * 길이 __len__ 
  * get item: index을 rgument으로 받아서 그 data 반환. 주로 dict 형식으로 반환.

* 주의
* 데이터 종류에 따라 함수가 달라야 함 NLP/CV
* 모든 순서를 한번에 다 할 필요 없다. 학습 직전 그때 그때 불러와서 transform 한다. cpu gpu 병렬처리 함. cpu에서 텐서변환 -> gpu 학습.

* 표준화된 처리 방법을 만드는 것이 데이터셋 모듈로 할 역할. 후속 연구에 아주 긍정적인 영향. 

* hugging face등으로 간단하게 만드는 것이 요즘 추세!

* 데이터로더 class
* 데이터셋은 1개를 어떻게 가져오냐
* 데이터로던는ㄴ 어떻게 묶어서 모델에 던지냐

* 학습 직전 텐서로 변환시키는 역할을 함. 
* (transform을 사용한)텐서 + batch으로 어떻게 묶을지 결정

* 에폭 1번 돌리면 이 전체가 한번.

* sampler
* batch_sampler
* collate_fn: 텐서 차원 변환 혹은 패딩해줄 떄. 함수 포인터처럼 함수 레퍼런스를 넘긴다




## Further Question
DataLoader에서 사용할 수 있는 각 sampler들을 언제 사용하면 좋을지 같이 논의해보세요!
데이터의 크기가 너무 커서 메모리에 한번에 올릴 수가 없을 때 Dataset에서 어떻게 데이터를 불러오는게 좋을지 같이 논의해보세요!


## custom dataset

* __init__에는 object init하니까 최대한 적게.

* dataloader가 데이터를 뽑을 때는 __getItem__에서 뽑는다. 그러니까 init은 적게. len을 뽑을 정도로만... 그래서 label 파일은 init에서 뽑아야 한다. init의 나머지는 다른 데이터 주소 등. trnsformers ojbect 받기 정도. 

* 나머지는 get item에서 한다. 그래야 데이터 로더 옵션에 따라서 부분적으로만 받아서 자원 절약 가능. 

## my_collate_fn
```
print(sample)
[{'X': tensor([0.]), 'y': tensor(0.)}, {'X': tensor([1., 1.]), 'y': tensor(1.)}]
```

## *args: positional arg. get as tuple.
* not only main use. but also for common functions. get multiple arguments at the same time and make it a tuple.

## **kargs: keyword arg. get as dict.

* zip(*iterable): unpack

* reduce: arguemnt: function and iterater.

* torch cat: 주어진 dimension을 유지하면서 쌓는다.

* torch stack: 별개로 쌓는다