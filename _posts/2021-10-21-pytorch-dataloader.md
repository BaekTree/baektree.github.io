---
title: "dataloader"
last_modified_at: 2021-10-13T16:20:02-05:00
categories:
  - pytorch
tags:
  - dataloader
---

# dataloader
## output dimension
* 데이터 1개의 tuple을 쪼개서 배치 단위로 concat한다. 

# 예시

## 자주 사용하는 일반적인 경우
* 데이터셋의 1개 example은 tuple이다.

```
class dataset
    ...
    def get item(index)
        ...
        return x, y

```

```
dataloader = Dataloader(dataset, ...)
x_batch, y_batch = next(iter(dataloder))
```

하나의 데이터 셋의 형식에... concat을 해서 내보냄. get item의 output이 (x,y)의 tuple 이라면, x와 y을 분리해서 각 x와 y들을 배치로 concat한다. (batch, x), (batch, y)으로 내보낸다.

## TensorDataset 사용하는 경우
TensorDataset을 사용해서 여러 텐서들이 tuple로 묶여서 들어간다고 해보자. (tensor1, tensor2, ..., tensor n).
TensorDataset의 get item은...

```
class TensorDataset
    def get item (index)
        return [ tensor[index] for tensor in tensors ]
```

이렇게 각 input tensor의 index 마다 tuple로 묶어서 하나의 example로 반환 함.
1개의 example은... (tensor1[i], tensor2[i], ..., tensorn[i]) 이렇게 됨.

이게 dataloader에 들어가면... 각 tuple의 내용을 분리해서 ... 배치끼리 concat을 한다.
( concat( tensor1[i], tensor1[i+1], ... ), concat( tensor2[i], tensor2[i+1], ... ), ... )