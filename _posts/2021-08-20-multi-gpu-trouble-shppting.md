---
title: "08-20-multi-gpu-trouble-shppting"
last_modified_at: 2021-08-20T16:20:02-05:00
categories:
  - boostcamp
tags:
  - camp
---
# torch mullti gpu

* 예전에는 데이터셋이 별로 없었음. 메모리도 작아서 어떻게 작게 넣어서 학습하ㄱ까

* 지금은 nvidea에서... gpu가 많다! 좋다! 많은 gpu 넣어서 많은 데이터 넣어서 학습해서 더 좋은 성능을 내자! multi gpu

* gpu가 2장 이상 이어야... Multi gpu가 됨... 우리한테는 1개만 준다... 그래서 개념으로만 설명을 해보게따...

* 용어
* single gpu
* multi gpu: 2개 이상

* 노드 / 시스템을 쓴다. 1대의 컴퓨터를 쓴다. 1개의 노드를 쓴다 = 1개의 컴퓨터 안의 1개의 gpu 씀 = single node single gpu

* single node multi pgu: 1개 커뮤터 여러개 gpu
* multi node multi gpu: 서버실. 다대다

* 두가지 방법
* 모델을 병렬화
* 모델을 뚝 가로로 자름. 반반 씩 다른 gpu에 넣는다
* 딱 알렉스 넷이다. 옛날: 메모리가 없어서. 지금: 크게 만들라고. 모델 병렬화가 새로운 연구 분야로 자릴 잡았다. 파이프라인에서 병목 현상 해결하기 등.
* 데이터를 병렬화
* 배치 잘라서 나눠서 넣는다. loss 합해서 평균한 다음에 전체에 미분.

* 알렉스 넷에서 중간에 gpu 교환 함. to(device) 뭐 이런 코드로...

* 모델 병렬화 문제
* 순차를 잘못하면 1개 돌 떄 필요한 정보 받느라 대기 하느라 기다림. 더 손해. 파이프라인을 만들어야 한다. 

* gpu 폭발 상태: 반장이 파업. 한 gpu에서 cordinate까지 해야 해서... 메모리가 조금 부족 함. 그래서 corndinate 하는 gpu는 다른 gpu들 보다 배치 크기를 조금 줄여줘야 한다. : 파이썬 DataParallel 모듈. 

* DistributedDataParellel: 모으는 작업이 없다. 각각이 ff와 bp을 각자 하고 나서 grad 결과만 cpu가 평균. 그래서 gpu 중 cordinator가 없다. 

* pin_memory = dram에 바로 데이터 올림. 데이터 처리가 빠르게 함. 

* 하이퍼 파라미터 튜닝
* 학습률
* 강세율
* 등등

* 옛날에는 손맛. 요즘은 좋은 도구들이 나왔다. 

* 원하는 결과가 안나왔을 떄
1. 모델을ㅇ 바꾼다 -> 이미 어떤 모델이 성능이 좋은지 대충 정리가 되었다. resnet / transformer
2. 데이터를 추가/변경 -> 이게 가장 중요
3. 하이퍼라마미터 튜닝 -> 유익이 생각보다는 크지 않다. 쥐어짜내기

* learning rate(NAS neural architecture ... auto ML)

* 마지막 0.01 쥐어짤 때 도저언! 요즘은 워낙 데이터가 많아서...

* 가장 자주 쓰는 법: grid searach or random search. 지금은 잘 안쓰고... 요즘은 베이지안 기반 기법들이 인기. 

* Ray 모듈이 유용하다. multi node nulti processig 지원 모듈. 하이퍼 파라미터 서치 기능 있음. 

* ahah 알고리즘으로 중간에 성능 안좋은 ㅁ하이퍼 파라미터들 cut out

* partial 데이터 자르는 모듈


* 토치 trouble shooting
* 많이 부딪히는 문제들

* out of memory
* 왜 어디서 발생했는지 알기 어려움
* error backtracking이 이상한데로 감

* 1차원 해결: batch size 줄여서 gpu 다시 시작해서 run.
이렇게 해결이 안되면 코드에 문제. 

* gpu util을 사용해서 상태를 확인

* GPUUtil.showUtilization()으로 메모리 확인. 메모리가 누수. 잘못쌓이고 있는 것을 확인 가능

* torch.cuda.empty_cache() fp 할떄 cache가 쌓임. bp에 쓰려고. 중단하고 다시 시작하면 쌓인거 클린.

* del tensorList하면 메모리에서 지우는 것? linked list을 끄는 것. empty_cuda()는 지금 당장. 이걸 학습 코드 맨 위에 둬라.

* requred_grad 하면 캐시를 기록해야 하고 computational graph 그려야 함. 메모리 잠식 중... 

* loss 합칠 때 loss를 텐서로 다 합치면 cache 메모리 그대로 가져 감. 값만 가져가기 위해 tensor.item()으로 파이썬 기본 value 객체만 옮겨라.

* 필요 없는 변수는 del으로 삭제하기. 가비지 컬렉터가 가지고 갈 수 있도록. 

* oom이 뜨면 배치를 1으로 해보고 해보기. 

* inference 시점에서 torch.no_grad 꼭 하기. tracking 하지 않아서 메모리 추가 사용 막자. 

* 다른 에러들
* cudann status not init 쿠다가 잘 안깔렸을 때

* device side assert 가 생기면 다양한 겅우. 검색 ㄱ

* 코랩에서 큰 사이즈 실행 하지 말자 lstm. 그냥 pretrain model 쓰세요. 

* cnn에서 입력 출력 크기 안맞는 경우 많다. torchsummary으로 사이즈 만들기. 

* float precision을 16 bit으로 줄일수도 있음. 메모리가 정말 없다면... 