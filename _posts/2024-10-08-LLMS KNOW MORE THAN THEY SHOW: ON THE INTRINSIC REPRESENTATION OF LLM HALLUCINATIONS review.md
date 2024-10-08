---
title: "LLMS KNOW MORE THAN THEY SHOW: ON THE INTRINSIC REPRESENTATION OF LLM HALLUCINATIONS review"
last_modified_at: 2022-01-19M16:20:02-05:00
categories:
  - nlp
  - paper-review
tag:
  - paper
  - NLP
---

# 이런 연구를 하시는 분들에게 추천: 

- 모델의 hallucination을 잡아야 하시는 분
- hidden representation을 사용해서 hallucination 오류를 개선하려고 시도하시는 분
- computing requirements: error detection에 LLM 하나와 소형 classifier 하나가 필요해요

# 배경: 
LLM에 할루시네이션이 많음. ‘틀린 것을 어떻게 찾아낼까’도 아주 중요함. error detection이 유망함. 생성된 결과물에서 에러를 찾는 여러가지 방법론이 있음. 주로 토큰의 representation의 mean을 측정하던지, 프롬프트의 마지막 토큰의 representation을 가져와서 이게 모델에 fit 한 질문인지 등 측정함. logit 꺼내서 확인하기도 함. 

# 가설: 
프롬프트에서 요구한 답변 GT에 해당하는 토큰들에 trustfullness 정보가 가장 많을 것이다. 따라서 이 토큰들의 hidden representation을 사용해서  true or false을 판단해보자. 

# 실험:

- 마지막 MLP의 output에서 hidden representation을 뽑았음.
- probe classifier의 경우에 sklearn의 logitstic regression을 classifier으로 사용함
- 학습데이터셋 모으기: 모델이 뱉은 답에 gold label있으면 해당 위치에 해당하는 hidden representatin 모음. open end question이 경우 gold label 없으면 LLM에게 시켜서 정답에 해당하는 토큰 extract하게 함. 그리고 위치 찾아서 hidden representation 뽑음. positive label 데이터로 삼아서 classifier에 학습함. negative example에 대한 말은… 논문에 나오지 않음. 아직 코드 공개 되지 않음. 일반적인 probe classifier와 유사하게 negative example을 모았을 수도 있음.
- 이 데이터들을 기존의 error detect하는 방법들에 적용함. ‘어떤 토큰을 사용하는가’가 주요한 변경점임. 기존: 끝 토큰, 토큰 평균 등등의 representation → LLM이나 gold label에 해당하는 토큰의 representation들.
- exact answer token에서 뽑은 representation이 다른 토큰 위치들보다 벤치마크 점수가 가장 높았음. 여러 방법들 중에서도 probe classifier가 성능이 가장 좋음. 하지만 특정 task에서 학습한 probe classifier는 다른 task에서는 동작하지 않음.

# 결론: 
exact match token에서 프롬프트에서 요구하는 알맞은 정답이 있는지 가장 많은 정보가 들어가있다. exact match token에  probing classifier 학습해서 error detection했더니 기존 연구 결과들보다 에러 찾는 성능이 좋아졌다. 이 방법이 모든 task에 대해서 동작하지는 않는다. 학습한 probing classifier와 유사한 task에서는 잘 동작하지만 task의 차이가 많아지면 동작하지 않는다. 

# 토의: 
- 정답에 해당하는 exact match token에 가장 많은 정보가 있다는 것은 사실 직관적으로 보이는데 여태 연구되지 않은 점이 조금 의아하다.
- probing classifier을 잘 학습하면, hidden state에서 특정 토큰에 대한 정오 여부를 판별할 수 있다. 이 내용을 자동화해서 reward model에 적용하면 특정 토큰 단위로 preference 점수를 줄 수 있지 않을까? 더 정교하고 섬세하게 HFRL task에 활용할 수 있을 것도 같다.
