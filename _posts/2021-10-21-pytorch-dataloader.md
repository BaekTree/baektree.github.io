---
title: "dataloader"
last_modified_at: 2021-10-13T16:20:02-05:00
categories:
  - pytorch
tags:
  - dataloader
---

# dataloader
dataset에 그냥 iterator을 sampler 등의 옵션만 해주는 것이다. 입력이 일반적인 (x,y) 페어이면 (batch, (x,y) ) 이렇게 뱉는다.
입력이 list이면 (batch, list) 이렇게 뱉나?

일단 들어가는 형식 그대로 배치만 묶어서 내보내는 것은 맞다. 하나의 데이터 셋의 형식에... concat을 해서 내보냄.
get item의 output이 (x,y)의 tuple 이라면, x와 y을 분리해서... (batch, x), (batch, y)으로 내보낸다.

TensorDataset을 사용해서 여러 텐서들이 tuple로 묶여서 들어간다고 해보자. (tensor1, tensor2, ..., tensor n).
TensorDataset의 get item은...

get item (index)

return [ tensor[index] for tensor in tensors ]
이렇게 각 input tensor의 index 마다 tuple로 묶어서 하나의 example로 반환 함.
1개의 example은... (tensor1[i], tensor2[i], ..., tensorn[i]) 이렇게 됨.

이게 dataloader에 들어가면... 각 tuple의 내용을 분리해서 ... 배치끼리 concat을 한다.