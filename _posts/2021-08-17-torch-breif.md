---
title: "0817-torch-breif"
last_modified_at: 2021-08-17T16:20:02-05:00
categories:
  - boostcamp
tags:
  - camp
---

# 토치
* 프레임워크

* 어려운 코딩을 쉽게.


* 토치 안쓰고 하나 하나 다 구현했다 옛날에는. 바닥부터.

* 다른 프레임워크 텐서플로우. 
* 자료가 많다. 관리 잘 됨. 오픈소스. 
* 구글 내부 도구 였다가 공개한 이유: 문서화를 잘 하기 위해서이다. 남들이 알게 하기 위해 오픈했다고 한다. 사실상 표준. 프레임워크 시대가 이미 도래.  프레임워크 공부 = 딥러닝 공부 ㄴ

* 다양한 프레임워크 많다. 리더는 두개. tf와 토치. 토치는 페북이 만들었다. 

## 케라스 토치 텐서
* 케라스는 wrapper이다. 껍데기. 하단에는 토치와 텐서로 구현. 사용하기 쉽게. high level API. 텐서가 2.0부터 쿠다 지원하면서 케라스랑 거의 통합 됨. 텐서 안에 있는 high level api. 케라스 프로젝트도 구글 내부에 있어서 그냥 통합. user friendly. 

## 토치
* low level api + high level 까지 가능. 속도 ㄱㅊ. 

## 차이: 
* computatinal graph. 미분 하려면 그래프로 표현해야 함. 
* tf: static 하게 그리고 실행할 때 실행. define and run.
* torch: 실행할 때 그림. dynamic computation graph.

## computational graph
* g = (x + y) * z

* tf. 그래프를 먽 ㅓ정의. 실핼할 때 데이터 feed = 데이터를 그래프에 넣는다.
* define by run: 실행을 하면서 그래프를 생성, dynamic comp gragh. DCG.

* 토치를 쓰면 디버깅하기 쉽다. 중간 중간 값 확인 가능. 

## 각 장점
* 텐서: production, cloud, tpu, multi gpu에 장점 + scalability
* 토치: 개발에서 디버깅ㅇ이 쉽다. 개발, 구현 등에 장점
  * 즉시 확인 가능 - 파이토낙
  * gpu 커뮤니티
  * 편하다
  * numpy + autograd + function
  * numpy 구조 가지는 텐서 객체
  * 자동 미분 연산 지원
  * dl 함수와 모델 지원함: 데이터셋, multi gpu, data augmentation 지원. 덜 복잡하게 할 수 있다. 

auto grad가 어떻게 동작하는가?에 대한 내용 면접 질문.  


## ML 프로젝트 with jupyter or vscode?
* 초기: 대화식. 디버깅 쉽다
* 공유와 재현, 재포 어렵다
* 실행순서 꼬인다 왔다 갔다
* 유지보수도 해야 한다
* 대안: 레고블럭처럼 OOP 와 모듈로 프로젝트 단위로 만든다.
* 다양한 템플릿이 있음
* 실행 데이터 모델 설정 로깅 지표 유틸 등

* 템플릿은 검색하면 잘 나온다

* 코랩을 로컬의 vscode에서 ssh으로 ...

* NGROK_TOKEN = '1mn46f5zchuMTvgUwv7MECrVYy1_7rVtSq7XzXgCd1qZB6CBE' # ngrok 토큰
PASSWORD = 'upstage' # 비밀번호 설정


vscode에 저장해둔 템플릿 코드 이해하기.

# viz
## text 
* 차트의 요소
  * 텍스트
  * 칼라
  * 멀티플
  * 정보

* 텍스트
  * 추가 정보
  * 오해를 방지
  * 과하면 간결함을 해친다
  * 적당히 써야 한다

* 요소
  * title
  * label: 축 텍스트
  * tick label; 축에 눈금. 스케일 정보
  * legend
  * annotation

https://colab.research.google.com/drive/1SgTsrwpwSaCQ1KF70BlwG9TPa3VbeMhA#scrollTo=MouRIqmo-_J7




## 특강: 좋은 개발자
1. 깔끔한 코드. 변경이 용이. 확장성. 유지 비용 낮은 코드. i,j 쓰지 좀 마라. tmp 좀 쓰지 마라. 
2. 적절한 논리력. 우리는 제한이 많다. 철저한 극단적인 논리는 극단적인 환경을 요구. 현실은... 웹 서버는 한계가 있다. 엔지니어링은 제약 사항에서 가장 적합한 솔루션을 찾아내는 것이다. 원리 탐색 능력. 단순한 디자인. 
3. 바꿀 수 없는 외부 요인. 그대로 둘 수 밖에 없다. 스스로 어떻게 지속 가능하고 성장 가능한 개발자가 될지 내적 요인을 발전시켜야 한다.
깔끔한 코드 작성법: acceptance test driven development. 테스트를 먼저 만들고 개발하라. 
* 사용하는 코드만 만들라
  * 안쓰는 코드들 다 지워버려라! 다른 사람의 시간을 낭비하지 마라. 미련없이 지워라. 혹시 나중에? 버전 컨트롤 시스템. 이러려고 쓴다.
  * 리팩토링: 외부에서 바꿔도 내부가 바뀌지 않도록 독립 독립 독립!
  * 코드 리뷰//
  * 단순하게 생각하라. 가장 단순한 경우 먼저 생각하라. 
  * 매일 몸 값 올리는 시간을 가져라. 동기부여를 꾸준히
  * 멀리 가려면 멀리 가라. 마음이 맞는 사람을 찾아라. 