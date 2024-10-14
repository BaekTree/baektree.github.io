---
title: RATIONALYST:-Pre-training-Process-Supervision-for-Improving-Reasoning-review
last_modified_at: 2022-01-19M16:20:02-05:00
categories:
  - nlp
  - paper-review
tag:
  - paper
  - NLP
---

# RATIONALYST: Pre-training Process-Supervision for Improving Reasoning 리뷰

Tags: review
Checkbox: No
문서 이름: RATIONALYST: Pre-training Process-Supervision
for Improving Reasoning

이런 연구자들이 보면 좋아요!

- 모델의 rationality 성능을 높이고 싶은 분
- inference 자원이 충분하거나 real time으로 추론이 필요없는 분: reasoning step 마다 candidate을 만들어서 가장 좋은 것 결정함. inference에 LLM 3개 필요해요.

배경

- LLM의 reasoning이 아직 imperfect하다.

가설

- LLM의 reasoning이 아직 imperfect한 이유는 LLM이 pre-training 단계에서 보는 데이터들의 reasoning이 imperfect하기 때문이다. 만약 더 명확한 reasnoing이 논리 단계 마다 드러나면 더 좋은 결과를 볼 수 있을 것
- rationalist 모델 하나 만들어서 사용자에게 생성하는 모델의 단계 단계 마다 explict한 reasnoing을 제공하면 생성 모델이 더 정확한 reasoning 결과를 줄 수 있을 것이다.

실험

- 개요
    - explict 한 reasnoing 추출해서 데이터 생성 후 모델을 학습하여 rationalist 만들기
    - 사용자의 프롬프트를 받으면 rationalist가 단계 마다 reasoning을 주고 LLM은 그 reasoning을 참고해서 다음 단계를 생성한다.
    - llama 3 instruct의 경우 이 방법을 사용했을 때 점수가 오름!
- rationalist 생성하기
    - 학습 데이터 만들기 + 학습하기
        - Pile 데이터 + 벤치마크 데이터를 raw data으로 사용한다.
        - reasnoing이 있는 데이터들만 뽑아내기 위해서 STS task으로 먼저 걸러낸다. 벤치마크 데이터와 유사한 데이터만 Pile에서 뽑는다.
        - base LLM으로 각 문서에서 rationale을 뽑아내게 한다. 뽑아낸 rationle이 모두 최종 LLM의 reasnoing에 도움을 주지는 않으므로 걸러낸다. 걸러내는 방식: rationale + GT 응답을 입력으로 주었을 때 다음 토큰마다 나오는 loss을 계산한다. rationale 없이 GT 응답을 다시 입력으로 주고 loss을 계산한다. 이때 토큰 단위로 rationale을 주었을 때 차이가 threshold 값 이상이면 rationale이 유의미하게 토큰 생성이 도움을 준다는 의미가 된다. 따라서 이런 rationale들만 살리고 나머지는 버린다.
        - 위의 과정으로 생성한 데이터로 LLM을 학습한다. 이 LLM은 unlabeled data가 입력으로 들어오면 implicit하게 있는 rationale을 생성해낼 수 있다. rationalist 탄생!
- 추론하기
    - 필요한 모델들
        - 사용자가 마주하는 LLM agent
        - rationlist을 만드는데 사용된 기본 base LLM
        - rationalist
    - 프롬프트 받음. Trajectory T에 추가
    - 반복
        - rationlist가 Trajectory T에 대한 rationale 생성.
        - llm agent가 trajectory 보고 다음 단계 reasoning candidates N개 생성
        - base LLM이 candidates 마다 타당성을 심사함. 심사할 때 rationle을 참고해서 가장 높은 타당성 점수를 가진 candidate을 Trajectory T에 넣음. 확정된 reasoining step이다.
        - stop condition이 나올때까지 반복함.
            
            ![image.png](assets/src/rationalistimage.png)
            
- 검증
    - 성능이 많이 뛰어요.
    - base LLM + 특정  벤치마크 fine tuning 보다 rationlist + base LLM의 성능이 더 좋아요.
    - 생성하고 있는 reasoning이 올바른지 심사하는 verifier의 task에서도 rationalist가 GPT4 보다 좋아요.
    - rationalist 학습할 때 그냥 벤치마크 데이터만 쓰는 것보다는 Pile 데이터에서 더 다양한 reasnoing 을 경험하게 하는 것이 rationlist 성능 향상에 좋아요.
    - 추론 시 explicit한 방법(llm agent가 다음 reasnoing 생성할 때 rationle을 함께 보고 생성함) 보다 implicit한 방법이 더 좋다. 저자들은 이 이유가 rationlist가 실수를 해서 이사한 rationle을 만들었을 때 explicit한 경우는 rationle을 같이 봐서 실수를 알아차리기 어렵다?
        
        ![image.png](assets/src/rationalistimage%201.png)
        

결론

- 모델을 직접 fine tuning 하는 것보다 general purpose reasoning llm으로 가이드를 주는 것이 더 성능이 좋아요.

토의

- 마지막으로 했던 task에서 아주 비슷한 연구를 했음. explicit한 rationle이 나오면 성능이 좋아진다. 논문으로 내지 못해서 아쉽다. 사람 생각하는 것 비슷비슷하다.