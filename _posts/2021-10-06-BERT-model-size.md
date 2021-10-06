---
title: "1006-BERT-model-size"
last_modified_at: 2021-08-18T16:20:02-05:00
categories:
  - NLP
  - Tokenizer
  - Preprocessing
---

버트 모델 크기는 512이다. 이것보다 긴 문장이 들어가면?

저도 그게 헷갈려서 버트 코드, torch nn.embedding, stack overflow 등을 뒤져봤는데, 결론적으로 고정이 아닌 것 같아요. 

(batch, max_len) -> nn.embedding(num_vocab, d_model) 을 통해서(batch, max_len, d_model)으로 바뀌면서 (batch, max_len)이 그냥 batch 처럼 병렬적으로 모델을 통과하는 것 같아요. 이게 embedding layer가 내부에 word2vec 처럼 임베딩 테이블로 구현되어 있어서 그런 것 같습니다… 그래서 심지어 config에 있는 model_max_len = 512을 넘어가도 warning은 내뿜지만 동작은 한다고 하네요… 코드를 보니까 단어 자체는 다 받아들일 수 있는데 position embedding에서 조금 성능이 하락할 여지가 있어 보이는데, 그 부분도 512을 원하는대로 늘리면 전혀 문제가 없어보였어요!

https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py
https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
https://github.com/huggingface/transformers/issues/1791