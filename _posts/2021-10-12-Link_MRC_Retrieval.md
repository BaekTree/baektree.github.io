---
title: "Link_MRC_Retrieval"
last_modified_at: 2021-10-12T16:20:02-05:00
categories:
  - boostcamp
  - nlp
tags:
  - MRC
  - Retrieval
---


#MRC와 Retrival을 연결
# Introduction to ODQA
지문이 주어지는 것이 아니라. 웹 전체 혹은 위키. 일단 문서를 뒤져야 함. 그 다음에 MRC 수행. 인풋와 아웃풋은 동일. 질문과 답변.

1999년도 부터 사실 문제가 제기 되었음.
기존 방법 순서
1. 질문에서 키워드 선택. 장소여야 한다. 나라여야 한다. rule based.
2. passage retrival: 연관된 문서를 뽑은 다음에 passage오 쪼갬. 그다음에 NER 등을 적용해서 feature으로 활용
3. classifier으로 어떤 passage가 정답이 있는지 분류해서 유저에게 보낸다.

현대는 3번이 조금 다름. 실제 답안 구간을 찾아서 답만 유저에게 보여준다. 

# ODQA의 방법 중 하나: 리트리버와 리더를 연결
ODQA을 푸는 가장 일반적인 방법론이다.


# 연구 동향
