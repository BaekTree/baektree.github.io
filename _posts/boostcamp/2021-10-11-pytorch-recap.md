---
title: "pytorch-recap"
date: 2021-10-11M15:34:30-04:00
categories:
  - recap
tags:
  - pytorch
  - recap
---

pytorch 정리

torch recap

# lr 정리
https://www.kaggle.com/isbhargav/guide-to-pytorch-learning-rate-scheduling

# custom model

huggingface에서 custom model을 만들때...
```
from transformers.modeling_utils import PreTrainedModel
class Mean_Pooling_Model(PreTrainedModel):
```
이렇게 해야... config.json이 모델이랑 동시에 저장된다. 그냥 nn.Module으로 하면 model만 저장되고, config.json은 저장 안됨. 실제 학습한 모델을 evaluate, predict 할때 정작 못쓴다....

# Expected object of scalar type Long but got scalar type Float for argument #2 'target' in call to _thnn_nll_loss2d_forward

https://discuss.pytorch.org/t/expected-object-of-scalar-type-long-but-got-scalar-type-float-for-argument-2-target/33102/2

```
labels = labels.to(device=device, dtype=torch.int64)

혹은... labels.long.to(device=device)
```

# scatter
torch.tensor.scatter(dim, index, src)

call하는 tensor에 적용함
src의 index에 해당하는 값을 tensor에 적용함
index는 src와 동일한 차원을 가져야 함

```
>>> src = torch.arange(1, 11).reshape((2, 5))
>>> src
tensor([[ 1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10]])
>>> index = torch.tensor([[0, 1, 2, 0]])
>>> torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
tensor([[1, 0, 0, 4, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 3, 0, 0]])
```

위 예시는... src의 값을 torch.zeros의 index에 차례로 끼워넣는다. 
index으로 들어온 텐서가 [[0,1,2,0]]이니까 src에서 첫번째 row만 해당이 된다. 그리고 4개만 있음. 따라서 src의 첫번째 row의 1,2,3,4가 torch.zeros의 0번째 row, 1번째 row, 2번째 row, 0번째 row에 적용된다. column은 항상 그대로 유지.

```
>>> src = torch.arange(1, 11).reshape((2, 5))
>>> index = torch.tensor([[0, 1, 2], [0, 1, 4]])
>>> torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)
tensor([[1, 2, 3, 0, 0],
        [6, 7, 0, 0, 8],
        [0, 0, 0, 0, 0]])
```

이 예시는 src에서 index의 자리에 해당하는 element들을 target에 넣는다. 1번째 row의 0~2의 element와 두번째 row의 0~2 element을 사용한다. target에 각각 넣을 위치가 index의 각 자리의 숫자이다. 각 자리를 넣을 때 dim = 1이므로 column으로 각각 넣는다. src의 1번째 row을 0번째 col, 1번째 col, 2번째 col에 넣는다는 의미이다. 그리고 src의 2번째 row의 값에서 0-2의 값들을, 2번째 row의 0번째 col, 1번째 col, 4번째 col에 넣는다. 



# torch dim

[1,2,3]: (3,)
일반적인 벡터. 1차원 벡터이다

만약 2차원 벡터라면?
[
    [
        1,2,3
    ],
    [
        4,5,6
    ]
]

2,3 행렬이다. 

한줄로 [[1,2,3],[1,2,3]]이렇게 나옴 보통.

(3,)과 (1,3)의 차이

(3,)은 1차원 벡터 [1,2,3]
(1,3)은 2차원 벡터 [[1,2,3]]