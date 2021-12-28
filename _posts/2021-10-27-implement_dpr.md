---
title: "implement_DPR"
last_modified_at: 2021-10-27T16:20:02-05:00
categories:
  - nlp
  - project
tags:
  - implementation
  - paper
  - NLP
---

hard negative는 batch 밖에서 bm25로 유사도가 높은 문서이지만 정답은 없는 문장이다. query 1개당 하나씩 만들어서 negative sample로써 현재 배치 전체에 동일하게 적용. 그래서 만약 배치 크기가 8이면, 8개의 새로운 negative 들이 각각의 example들에 동일하게 적용. 따라서 배치 내부에서 중복되는 것이 없어야 한다. 한 query의 negative가 다른 query의 positive가 되지 않도록 구성해야 한다.

따라서 dataloader에 들어가기 전에 manually random하게 데이터 묶음을 만든다. 그리고 데이터로더에서는 shuffle을 꺼야 함

모델에 들어가서는? 
모델 안에 들어가서 어떻게 됨?

미니 배치에 16개의 context가 있음.
8개는 truth. 8개는 hard negative.

query는 8개가 들어옴.

context가 모델에 들어가면...
`(16, max_len) -> (16, max_len, emb) -> cls -> (16, emb)`
여기서 pos, neg, pos, neg 이렇게 들어간다.

query가 들어가면...
`(8, max_len) -> (8, max_len, emb) -> cls -> (8, emb)`

`context.T -> (emb, 16)`

dot product
`(8, emb) * (emb, 16) -> (8 * 16) -> softmax -> (8 * 16)`
diagonal이 ground truth이다? 아니다. row * 2번째 col 값이 prediction이다.

그러면 dot product 한 다음에 코드를 어떻게 짜냐면...
`dot_prod = torch.matmul( q_emb, p_emb.T ) # ( 8 * 16 )`

dot_prod  

use gather: 
```py
indices = torch.zeros(batch_size, batch_size + batch_size * num_hard_neg) # ( 8 * 16 )
for i in range(indices.size(0)):
    indices[i][i*2] = 1

torch.gather(dot_prod, 1, indices)
```