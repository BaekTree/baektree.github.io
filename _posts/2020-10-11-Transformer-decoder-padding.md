---
title: "2021-10-11-Transformer-decoder-padding"
date: 2021-10-11T15:34:30-04:00
categories:
  - NLP
tags:
  - Transformer
  - padding
  - decoder
---

# Transformer-decoder-padding

* Transformer으로 입문을 해서 transformer의 decoder이라고 제목을 달았지만 대부분의 decoder 모델에 동일하게 적용되는 작용일 것임.

## Q: padding을 먹일 때 이 padding에 대한 decoder output은 어떻게 될까?

(batch, dec_len) -> nn.embedding -> (batch, dec_len, d_model) -> masked-self-attention -> Q dot K.T -> (batch, num_head, dec_len, dec_len) -> 아래 row들이 padding에 해당한다 -> softmax(Q dot K.T) dot V -> (batch, num_head, dec_len, d_k) -> concat -> (batch, dec_len, d_model) -> residual connection and linear norm -> attention with encoder memory -> Q dot K.T -> 아래 row들이 padding에 해당한다 -> (batch, num_head, dec_len, enc_len) -> softmax(Q dot K.T) dot V -> (batch, num_head, dec_len, d_k) -> concat -> (batch, dec_len, d_model) -> linear transformation -> ... -> linear transformation(d_model, t_vocab) -> (batch, dec_len, t_vocab) -> softmax -> arg max for each dec_len -> Cross Entropy Loss -> compare ground truth and and predictoin.


* During training: 
  * we will use padding and masking to make all sample sequence sizes equal.
* During inference: 
  * For variable output sequence size, we will modify the decoder such that it will work in a loop and stop generating output when a condition is met. This condition can be in two ways:
  * If the decoder generated the maximum number of outputs defined by the user, or
  * If the decoder generated a special “STOP” symbol


pad가 들어가면 항상 (dec_len, *)에서 아래 row들이 padding을 담당한다. mask가 입혀질 때 transpose되어서 column에 적용 됨. 따라서 masked 된 부분은 softmax에서 0이 된다. 아래 padding 되는 부분은 살아서 ... ground truth padding이 되도록 학습되어야 한다. 

그러면 tunz에서 처음 본 사람은 왜... mask된 것을 잘랐지? 그 사람이 구현한 것은 language modeling일 텐데? 위에 적은 대로라면 language modeling에서 적용하면... language modeling에 decoder가 필요한가 그런데? 

그...  그러면 이 아저씨가 잘못 해석하고 있나? 아니면... 이 아저씨는 seq2seq을 말하고 있는데... 그러면 seq2seq을 말하는 google tensorflow는 어떻게 하고 있나? 

여기도 무시 함.

Reference
https://medium.com/deep-learning-with-keras/seq2seq-part-e-encoder-decoder-for-variable-input-output-size-with-teacher-forcing-92c476dd9b0



https://discuss.huggingface.co/t/encoder-decoder-loss/4335
https://github.com/pytorch/fairseq/issues/1431

여기에 따르면 진짜로 무시 함. ground truth가 pad이면 무시 함. 

결정적인 힌트: tensorflow MNT task example에서 padding을 무시 함.
https://www.tensorflow.org/text/tutorials/transformer#loss_and_metrics

한글 문서: 블로그로 잘 정리해두셨다.
http://incredible.ai/nlp/2020/02/29/Transformer/

harvard: label smoothing에서 padding을 잘 정리
https://nlp.seas.harvard.edu/2018/04/03/attention.html#loss-computation


근데 이렇게 padding을 무시해도 되나? 올바로 학습을 할 수 있을까?

(batch, len, d_model) 이렇게 모델에 들어 감. 
decoder가 들어갈 때 d_model만 중요. batch, len은 그냥 배치 처리. d_model이 pad일 때도 있고 실제 의미 있는 토큰일 수도 있음. 모델은 encoder에서 앞뒤 단어들과 attention 한다. decoder에서 앞 단어들과 attention 한다. 각 example 마다 앞 단어들의 수가 각각 다르다. 긴 문장이면 더 많은 앞 토큰들이 있다. 

이런 상황에서 패딩을 학습한다?는 의미는... 패딩이 들어갔을 때 output에 도달해서 Padding인 ground truth와 비교해서 loss을 계산한다. 그 값에 대해서 gradient descent을 통해서 파라미터를 업데이트 한다는 것이다. attention을 하고 있으므로 앞 단어들에 대해서 

# BERT?
BERT 같은 인코더 기반에서는 pad을 어떻게 처리 하나? 여기도 loss을 계산할 때 padding을 제외할까? 일단 MLM에서는 해당 masekd된 위치만 확인한다. 그래서 padding을 확인 안함.

next sentence predictoin에서는? 이 task의 목적은? 다음 sentence을 예측하는 것. labeled되어서 true or false 확인하는 것. CLS만 확인한다. 그래서 마찬가지로 padding을 loss에서 확인 안함. 

다른 fine tuning은? 전에 만든 ppt을 봐야겠다. 