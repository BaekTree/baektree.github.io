---
title: "YOLO Net"
last_modified_at: 2021-04-01T16:20:02-05:00
categories:
  - deep-learning
  - paper-review
  - cv
---
# Paper_Review_YOLO_NET

* Ng이 읽기 어려웠다고 했다. 친구들한테 물어봤더니 친구들도 어렵게 읽었다고 했다. 디테일이 무슨 내용인지 알기 어려워서 논문 오픈소스도 찾아보고 했다고 한다. 우리가 읽을 때에도 어렵더라도 괜찮다고 했다. 
* 말하려고 하는게 뭔지 알고 있으니까 그래도 더 쉽게 받아들일 수 있는 것 같다. 모든 논문이 마찬가지인 것 같다. 말하려는게 뭔지 직관적으로라도 감을 잡고 있으면 디테일을 더 쉽게 이해할 수 있는 듯.

## 읽기 시작!
# Iteratoin 1
Title
You Only Look Once: Unified, Real-Time Object Detection

Abstract
* Prior work on object detection repurposes classifiers to per- form detection.
* we frame object detection as a re- gression problem to spatially separated bounding boxes and associated class probabilities.
* A single neural network pre- dicts bounding boxes and class probabilities **directly from full images in one evaluation.**
  * regression으로 detect 한다고?
  * 처음 읽었다면, bounding box가 뭔지도 헷갈렸을 듯.
  * directly from full images in one evaluation.이 부분이 무슨 말인지... detail이 필요하다.

* 성능:
* Our unified architecture is extremely fast
* YOLO makes more localization errors but is less likely to predict false positives on background.
* YOLO learns very general representations of objects. It outperforms other de- tection methods, including DPM and R-CNN



figures
resize and pridict
B and P
architecture

아직 뭔소린지 모르겠다... ㅎ

# Iteration 2
Introduction

Systems like deformable parts models (DPM) use a sliding window approach where the classifier is run at evenly spaced locations over the entire image [10].

More recent approaches like R-CNN use region proposal methods to first generate potential bounding boxes in an im- age and then run a classifier on these proposed boxes. After classification, post-processing is used to refine the bound- ing boxes, eliminate duplicate detections, and rescore the boxes based on other objects in the scene [13]. These com- plex pipelines are slow and hard to optimize because each individual component must be trained separately.

This unified model has several benefits over traditional methods of object detection.
First, YOLO is extremely fast. Since we frame detection as a regression problem we don’t need a complex pipeline. This means we can process streaming video in real-time with less than 25 milliseconds of latency. Furthermore, YOLO achieves more than twice the mean average precision of other real-time systems

빨라서 초당 25프레임 비디오에도 사용 가능. 

Second, YOLO reasons globally about the image when making predictions. Unlike sliding window and region proposal-based techniques, YOLO sees the entire image during training and test time so it implicitly encodes contex- tual information about classes as well as their appearance. 

잘린 구간에 들어오면 에러가 높아지는 것을 개선했다고 말하는 듯. sliding window는 잘라서 예측. YOLO는 전체 이미지 받아서 학습하고 예측. 그래서 context을 파악할 수 있다? Fast R_CNN도 sliding window을 쓰나? 

Fast R-CNN, a top detection method [14], mistakes back- ground patches in an image for objects because it can’t see the larger context. YOLO makes less than half the number of background errors compared to Fast R-CNN.

Third, YOLO learns generalizable representations of ob- jects. When trained on natural images and tested on art- work, YOLO outperforms top detection methods like DPM and R-CNN by a wide margin.

예술에 사용했을 때, 다른 도메인에서도 성능이 좋았다. 

YOLO still lags behind state-of-the-art detection systems in accuracy. While it can quickly identify objects in im- ages it struggles to precisely localize some objects, espe- cially small ones.

성능이 그래도 state of art 보다는 조금 떨어짐. 특히 작은 물체들에 힘들어 한다. 그래도 배경은 더 잘맞추더라.



figures

conclusion

a unified model for object detec- tion. sliding window을 잘라서 학습하는게 아니라 한번에 한다구... unified 인것 같다. 

can be trained directly on full images. 

빠르다. fast YOLO는 더 빠르다. generalization에 좋다. 

# Iteratoin 3
overall reading with fast speed

2. Unified Detection

sliding window와 차이점 때문에 unified가 먼저 오는 듯. 가장 중요한 단어가 unified인가... We unify the separate components of object detection into a single neural network. Our network uses features from the entire image to predict each bounding box. It also predicts all bounding boxes across all classes for an im- age simultaneously. This means our network reasons glob- ally about the full image and all the objects in the image. 

잘라서 학습하는게 아니라 전체 한번에 학습 함. location도 bounding box으로 안다. class도 구분한다. 동시에. globally understands the picture.  

Our system divides the input image into an S × S grid. If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object.

S,S으로 나눈다. object 중심이 어떤 grid에 있으면 그 grid가 classfy할 책임이 있음.

IOU으로 confiddence 판단. 

prediction: 
x,y는 중심 위치. 
w, h는 가로 세로 길이. 실제 길이의 비율로 표현.
그리고 confidence C. class probability. P(class|object)

한 grid cell에 1개의 class만 판단. softmax인가 보지? 

각 주어진 object의 조건부 확률 class. 베이즈룰 곱해서 p(class) * IOU으로 각 class의 기대 prediction을 파악 가능? 각 p(object)은 1개의 grid cell에만 종속되어 있음. 다른 grid 와는 별개. 


2.1. Network Design

Our network architecture is inspired by the GoogLeNet model for image classification [34]. Our network has 24 convolutional layers followed by 2 fully connected layers. Instead of the inception modules used by GoogLeNet, we simply use 1 × 1 reduction layers followed by 3 × 3 convo- lutional layers, similar to Lin et al [22]. The full network is shown in Figure

GoogLeNet을 조금 참조해서 1 by 1 사용함. 근데 inception module을 쓴건 아니고 그냥 Lin의 아이디어 처럼 1 by 1으로 spare하게 만듦. overfitting 방지. nueron colleratio 사용. 

fig3 을 보면 layer block 마다 1 by 1 * channel을 사용해서 bottle neck을 만들고 다시 확장시키고 있음. 

1 -> 3 -> 1 -> 3 -> poll -> ... 반복

2.2. Training

We pretrain our convolutional layers on the ImageNet 1000-class competition dataset

For pretraining we use the first 20 convolutional layers from Figure 3 followed by a average-pooling layer and a fully connected layer.



Ren et al. show that adding both convolutional and connected lay- ers to pretrained networks can improve performance [29]. Following their example, we add four convolutional lay- ers and two fully connected layers with randomly initialized weights.

Our final layer predicts both class probabilities and bounding box coordinates.

We normalize the bounding box width and height by the image width and height so that they fall between 0 and 1.

We parametrize the bounding box x and y coordinates to be offsets of a particular grid cell loca- tion so they are also bounded between 0 and 1.

We use a linear activation function for the final layer and all other layers use the following leaky rectified linear acti- vation:

We use sum-squared error because it is easy to op- timize, however it does not perfectly align with our goal of maximizing average precision. It weights localization er- ror equally with classification error which may not be ideal.

min square error을 처음 사용. optimize 하기 쉬윈까. 그런데 이 loss는 location 에러와 class 에러를 동일하게 비중을 두었음. 

Also, in every image many grid cells do not contain any object. This pushes the “confidence” scores of those cells towards zero, often overpowering the gradient from cells that do contain objects.

많은 grid가 object을 포함하지 않음. 따라서 IOU = 0이 됨. 근데 이렇게 하면 그래디언트할 때, IOU 0이 미치는 영향이 너무 커서 실제 object 있는 cell의 gradeint가 적용되지 않음. 

그래서 가중치를 적용함. 람다 coord와 람다 noobj이다. 0.5와 0.5으로 적용함. 

Sum-squared error also equally weights errors in large boxes and small boxes. Our error metric should reflect that small deviations in large boxes matter less than in small boxes. 큰 박스에서 조금 차이가 난 것은 작은 박스에서 같은 차이가 난 것보다 영향력이 훨씬 작아야 함. 그래서 루트를 씌웠음. 큰 영향이 작게 나타나도록. 

OLO predicts multiple bounding boxes per grid cell.

bounding box prediction은 location이다. P(object)을 예측. 

    At training time we only want one bounding box predictor to be responsible for each object.


박스의 수도 정한다. 하이퍼 파라미터. 그리고 어떤 박스가 responsible한지 알아야 한다. 뭐 2개라면 좌우에 하나씩 하나? 그러면 test set에 미리 표시해두나? 이 object는 몇번째 박스가 맡아야 한다. 

cost: 박스가 못맞출 때 올린다. class 틀리면 올린다. 미리 할당해두지는 않는 듯. summaion이 동시에 붙어있는 것을 보니까. 무튼 현재 eg에 대해서, 그리고 각 grid에서, responsible한 박스가 위치를 틀린 만큼 cost 증가. 길이 틀린 만큼 cost 증가. 

에러: object가 

output 이미지의 각 cell에 임의 개의 anchor boxex가 있다. 각 박스는 object가 들어있을 확률 p_c, obj의 중심 위치, bx, obj의 크기 bh, hw에 대한 예측 값, 그리고 각 class 예측 값을 가진다. class 예측 값은 가장 큰 class의 argmax을 가져서 실수일 수도 있고, 각 class의 예측 확률로 벡터일 수도 있다.

assignment을 하면서 알게 된점

p_c는 각 박스에 obj가 있을 확률 예측.
c_i는 class 예측 확률.
obj가 있고 c_i일 확률 : P(c_i) * P(O).

이걸로 박스들 중 가장 큰 물체와 class을 판단하는 것. 그리고 역전파. 

사용한 함수
C1 tf.argmax(tensor), tf.reduce_max(tensor), tf.boolean_mask(tensor, filter) where filter = [...] < real_number
C2
C3
C4


yolo2 보기 전에 fastRNN의 regional propose을 보기.