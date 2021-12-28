---
title: "0818_autograd"
last_modified_at: 2021-08-18T16:20:02-05:00
categories:
  - project
tags:
  - camp
  - autograd
  - pytorch
---

## autograd of Torch

1. 텐서의 모든 연산에는 autograd을 상속한다
2. . torch.autograd 객체는 forward와 backward 함수를 가진다.
3.  autograd는 forward할 때, context 자료구조를 써서 어떻게 forward을 계산하는지 기록해둔다. 이것이 자주 등장하는 tracking이다.

4. 파라미터를 설정할 때 required grad = true라고 하면 텐서의 grad pp와 grad_fn pp가 활성화 된다. 이때 grad는 미분 값을 저장하는 곳. grad_fn은 autograd의 미분 함수가 저장되어 있다. forward을 하면서 주어진 텐서 연산을 따라서 grad_fn을 저장해두었을 것이다.
5. tensor.backward()을 하면 grad_fn의 함수를 실행한다. 
6.  backward을 실행하면 이 값을 가지고 와서 앞에서 전달받은 미분값과 함꼐 계산해서 현재 텐서의 grad을 계산해서 grad pp에 저장한다. 그리고 context로부터 이전 노드로 미분값을 전달하면서 이동한다. 
7. backward가 끝나면 각 텐서의 grad pp을 사용해서 파라미터를 업데이트 한다. 이떄는 no_grad을 해야 함. 트래킹 되지 않도록.
8. 각 텐서의 grad pp을 초기화ㅎ야 함.

```
a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y using operations on Tensors.
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    loss = (y_pred - y).pow(2).sum()
    loss.backward()

    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # Manually zero the gradients after updating weights
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None
```

1. 이 과정이 nn 모듈에서도 동일하게 동작. nn 모듈의 각 모듈들 역시 텐서와 파라미터로 이루어져 있다
2. layer와 block을 초기화할 때 required grad = true로 해서 파라미터를 만든다
3. 모델에서 forward 함수를 만든다
   1. 사용되는 각 모듈들의 forward의 연산 역시 autograd을 참조한다. 따라서 관련한 미분함수를 연산되는 텐서의 grad_fn에 저장.
4. 결과로 나오는 loss 함수는 텐서를 반환. loss 함수 역시 module을 상속하고 모든 module의 forward와 backward는 autograd을 참조.
5.  print(loss)을 살펴보면 텐서를 반환하고 grad_fn으로 함수 레퍼런스를 가진다.
6.  파라미터로 학습 가능한 각 텐서들은 grad property을 가지고 현재 단계의 gradient 값을 저장한다.
7.  .  loss.backward()을 하면 loss에 해당하는 텐서의 grad_fn에 해당하는 함수가 실행되고, 텐서의 grad property = grad + new grad으로 업데이트 된다. 
8.  autograd의 backwrad 함수는 forward할 떄 미리 저장해둔 context에서 값을 받아와 grad을 계산한다.

* 모듈이 autogard을 참조?
https://discuss.pytorch.org/t/custom-loss-autograd-module-what-is-the-difference/69251