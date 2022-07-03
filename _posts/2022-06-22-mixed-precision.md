---
title: "2022-06-22-mixed-precision"
last_modified_at: 2022-06-22T16:20:02-05:00
categories:
  - DeepLearning
tags:
  - DeepLearning
  - pytorch
---

# Pytorch mix precision

토치에 hapf precision 이라는 기능이 있다. 

### 왜 쓰나?

텐서 타입을 32바이트에서 16바이트로 바꿔서 속도와 메모리 정확도도… 속도와 메모리는 그렇다고 쳐도 정확도는? 정확도가 좋아지는 이유는?

엔비디아에서 발표했다.

### 문제 정의와 해결법

16바이트로 줄이는 것은 그냥 dtype float16으로 줄이는 것으로 할 수 있다. 그러면 그냥 속도도 빠르고 메모리도 적어지니까 좋겠네? 그러면 그냥 줄이면 되나? 아니다. 16개의 자리 보다 32개로 표현할 수 있는 것이 더 많다

그래서 문제가 발생한다. 경사하강법을 쓸 때 미분 값이 작으면 점점 역전파되는 값이 점점 작아지고 0에 수렴함. 업데이트가 되지 않는다. 그냥도 이런데… 32바이트에서 16으로 줄이면 특정 작은 범위의 32 값이 다 0에 가까워진다. 0에 가까운 수들이 더 많아짐. 그래서 역전파 소실이 발생한다. 이를 해결하기 위해서 32와 16을 같이 사용하는 아이디어를 제시했다. 

### 원리

일단 32짜리 모델 파라미터가 있다. 그러면 속도와 메모리를 위해서 전파되는 값을 16으로 낮춰서 네트워크를 통과시킨다. 그러면 전파 빠르게 됨. Loss에 도달했다. 그런데 위에 적은 것처럼 16바이트에는 값이 0에 해당하는 값이 많아져서 역전파 소실이 발생할 경우가 많아진다. 이걸 방지하기 위해서  loss의 여기서 값을 다시 조금 키운다. 여기서 특정 상수 만큼 곱해줌. 이를 loss scaling 혹은 그래디언트 스케일링이라고 부른다. 값을 키우고 계속 16 바이트를 유지하면서 역전파. 각 위치에 도착하면 32로 다시 키워서 32짜리 가중치에 업데이트한다. 

더 구체적으로 설명?

- 매 iteration 마다, 32 bit의 weight 모델의 copy을 만든다. 이 copy weight을 16비트로 바꾼다. 그리고 forward 및 back propagation에서 쭉 이 copy를 사용할 것이다.
- copy한 16비트 weight을 통과시켜서 forward propagation을 한다.
- 나온 값으로 loss을 구한다. loss을 계산한다.
- loss에서 나온 값으로 weight에서 back propagation으로 gradient을 구한다. 이때도 그대로 16비트를 유지한다. gradient을 구하면 각 텐서에 grad value 값을 저장할 것이다.
- optimizer에서 step을 해서 grad 값을 텐서 값에 업데이트할 때, 각 grad 값을 32비트로 변환한다. 그리고 원본 32비트 weight에 lr과 함께 곱해준다. 여기서 32비트로 다시 옮기는 이유는 일반적으로 매우 작은 learning rate 때문이다. 값이 매우 작아져서 lr과 곱해진 grad 값도 작아진다. 그런데 16비트 특성 상 많은 범위가 0이 되어버림. 이것을 방지하기 위해서 32비트로 다시 돌려서 세밀하게 표현하여 0이 아닌 값이 weight에 업데이트 되도록한다.

그런데 16비트이기 때문에 lr을 곱하기 전에서 grad 값 자체가 0에 가까워져 버리는 경우가 있다. 이 문제를 해결하기 위해서 grad 값 자체를 키워준다. 이것을 loss scale이라고 부른다. 16비트 weight을 통해서 16비트 loss가 나오면, grad 값이 너무 작아지지 않도록 특정 값을 곱해준다. 그리고 각 weight에서 grad을 구한다. 그리고 이 scale 상수로 다시 나눠준다. chain rule에 의해서 한번 값이 작아지면 연쇄적으로 작아지게 하는 것이 아니라 작아지는 특정 weight에만 적용되도록 한것이다. 이 과정까지 포함해서 전체 과정을 다시 쓰면 다음과 같다.

- 매 iteration 마다, 32 bit의 weight 모델의 copy을 만든다. 이 copy weight을 16비트로 바꾼다. 그리고 forward 및 back propagation에서 쭉 이 copy를 사용할 것이다.
- copy한 16비트 weight을 통과시켜서 forward propagation을 한다.
- 나온 값으로 loss을 구한다. loss을 계산한다.
- loss에 scale 상수를 곱한다.
- loss에서 나온 값으로 weight에서 back propagation으로 gradient을 구한다. 이때도 그대로 16비트를 유지한다. gradient을 구하면 각 텐서에 grad value 값을 저장할 것이다.
- grad 값에 scale 상수로 다시 나눠준다. 작아지는 결과가 해당 weight에만 적용되고 연쇄적으로 적용되는 것을 막아준다.
- optimizer에서 step을 해서 grad 값을 텐서 값에 업데이트할 때, 각 grad 값을 32비트로 변환한다. 그리고 원본 32비트 weight에 lr과 함께 곱해준다. 여기서 32비트로 다시 옮기는 이유는 일반적으로 매우 작은 learning rate 때문이다. 값이 매우 작아져서 lr과 곱해진 grad 값도 작아진다. 그런데 16비트 특성 상 많은 범위가 0이 되어버림. 이것을 방지하기 위해서 32비트로 다시 돌려서 세밀하게 표현하여 0이 아닌 값이 weight에 업데이트 되도록한다.

### 어떻게 사용하나? pytorch

크게 두가지를 해줘야 한다. 1. 전파할 때 32을 16으로 바꾸기. 2. Loss 값에 스케일 업을 해줘서 그래디언트 구하고 역전파하기. 

파이토치는 자동으로 이것들을 다 해주는 기능이 있음. Auto cast와 scaler이다. 

- autocast
    - Auto cast는 concext manager을 선언하면 해당 scope 안에서 전파되는 텐서를 다 16으로 바꿔준다.
- grad_scaler
    - 그리고 나오는 loss 값을 scale up 하기 위해서 scaler 에 loss 을 넣어준다. 그리고 backward와 optimizers step으로 업데이트.

```python
model = net().cuda()
optimizer = optim.SGD(model.parqmeters(),...)

scaler = grad_scaler()# scale up을 해주는 scaler!

for e in epochs
for input, label inn data
    with autocast # autocast scope 안의 tensor 값을 16 바이트로 줄여준다.
        prediction = model(input)
        loss_value = loss(preduction, label)
    optimizer.zero_grad()
    scaler(loss_value).bawkward()# scaler 안에 넣고 값을 키워준다음에 그 값을 기준으로 역전파
    scaler(optimizer).step()# scale up 된 그래디언트 값을 가지고 update
    scaler.update()# scale 값을 업데이트

    
```

scaler class을 더 뜯어보았다.

- init scale
    - 처음 시작하는 scale 값!
- backoff_factor
    - 16비트는 표현할 수 있는 수가 적어서 특정 값을 넘어가면 inf 혹은 NaN이 계산된다. 이 값을 weight에 update하면 쓰레기 값이 입력되어서 망함… 그래서 scaler에서 step을 할때 grad 값이 inf 혹은 NaN인지 확인한다. 만약 이런 쓰레기 값이 발생하면, 값이 너무 커서 발생한 경우이므로, 줄여줘야 한다. 이때 scale 값에 backoff_factor만큼 곱해줘서 scale 상수 값을 줄인다. 그래서 이 backoff_factor은 0보다 작아야 한다.
- growth_interval
    - 이 인터벌의 step 동안 inf, NaN이 연속해서 발생하지 않으면 growth_factor 만큼 scale 값을 올린다.
    - 값이 충분히 작아서 inf, NaN이 나오지 않으므로 0에 가까운 값이 자주 나오는 상황일 수 있다. 그래서 미분 값 소실이 발생하지 않도록 값을 키워주는 것이다.
- growth_factor
    - growth_interval 동안 16비트를 초과하는 값이 나오지 않으면 이미 충분히 작은 값이라서 미분 값 소실이 발생할 수 있다. 그래서 이 값을 scale을 통해서 미분 값들을 키워준다!