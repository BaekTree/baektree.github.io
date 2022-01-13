---
title: "huggingface-recap"
date: 2022-01-13T15:34:30-04:00
categories:
  - recap
tags:
  - huggingface
  - recap
---

# 인코딩 값을 되돌리기
## decode
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ids = tokenizer("Father enters the room")['input_ids']
tokenizer.decode(ids, skip_special_tokens=True)
```


ref
huggingface tutorial
huggingface tokenizer documemnt: https://huggingface.co/docs/transformers/main_classes/tokenizer

### decode() list[int]만 받는다. 배치 단위로 바꾸려면 batch_decode 사용
```python
for batch in train_loadera:
    ids = batch['more_toxic_ids'] # (B, max_seq_len)
    print( tokenizer.batch_decode(ids, skip_special_tokens = True) )
```

* 기본 토크나이저 클래스는 tokenization_utils.py 의 PreTrainedTokenizer(PreTrainedTokenizerBase) class이다. 자주 쓰는 roberta는 이 순서로 상속 함. 
roberta -> gpt tokenizer -> PreTrainedTokenizer -> PreTrainedTokenizerBase으로 상속. PreTrainedTokenizer에 _decode 함수가 있고, 사용은 PreTrainedTokenizerBase의 decode 함수를 사용한다. _decode는 list[int] 만 받는다. 그래서 배치를 한꺼번에 되돌리려면 batch_decode을 쓰면 됨.

https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/tokenization_utils.py

# wandb table

```python
# assume a model has returned predictions on four images
# with the following fields available:
# - the image id
# - the image pixels, wrapped in a wandb.Image()
# - the model's predicted label
# - the ground truth label
my_data = [
  [0, wandb.Image("img_0.jpg"), 0, 0],
  [1, wandb.Image("img_1.jpg"), 8, 0],
  [2, wandb.Image("img_2.jpg"), 7, 1],
  [3, wandb.Image("img_3.jpg"), 1, 1]
]
          
# create a wandb.Table() with corresponding columns
columns=["id", "image", "prediction", "truth"]
test_table = wandb.Table(data=my_data, columns=columns)
```

ref
https://huggingface.co/docs/transformers/main_classes/tokenizer