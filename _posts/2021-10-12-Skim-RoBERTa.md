---
title: "Skim-RoBERTa"
last_modified_at: 2021-10-12T16:20:02-05:00
categories:
  - nlp
  - paper-review
tag:
  - RoBERTa
  - paper
---


# Skim-RoBERTa
- RoBERTa(https://arxiv.org/pdf/1907.11692.pdf)
    - Dynamic Masking 사용
        - 에폭 마다 masking을 다르게 준다.
    - NSP을 뺐음
        - Downstream 성능이 더 잘나온다는 실험을 함... 논란 중
    - Data을 아주 아주 많이 학습
    - Byte pair encoding
    - 4가지 해서 SOTA 달성