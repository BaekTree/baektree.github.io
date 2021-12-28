NLP: 자연어처리
NLG: 자연어 생성

문단 일부를 보고 다음 단어를 예측
machine trainslation
question answering
document classification

학회
ACL
EMNLP
NAACL

문장의 정보: 단어들.
토큰들.
문장은 토큰들이 sequence으로 이루어짐
문장을 단어로 쪼개는 것이 토크나이징
어미도 변한다. 맑다.맑은데. study. studying. 이들의 변화 속에 같은 단어라는 것을 컴퓨터가 알아야 함. 

steimg: 단어 어근의 의미를 추출하는 작업. 

가장 low level의 task이다.

2.word level task.
NER: 고유명사 인식 task. 뉴욕타임즈가 세 단어가 아니라 신문사 하나로 인식해야 함. POSpart of speech. 문장 성분을 인식하는 task. 어떤 문구가 형용사이고 명사인지 인식

3.문장 level task. 감정 분석. this movie is not that bad. 긍정문장으로 분류. 기계 번역. 문장 전체를 이해하고 단어별로 변환해야 함. 문법과 어순도 고려해야 함.

4.문단 level task. Entailment prediction. 두 문장간에 논리 등을 고려. 두 문장이 논리적으로 타당하게 연결되어 있는가? 두 문장이 논리적 모순을 가지고 있지는 않은가? QuestionAnswer: 검색 결과를 나열하는 것이 아니고 답을 검색 결과 맨 위에 나타내주는 구글. 문서들을 죽 검사하고 독해를 통해 질문에 대한 정답을 알아내서 사용자에게 제시. dialog system.. 챗봇처럼 대화. summarization: 뉴스 문서를 한줄요약 등. 

text mining: 실시간 트렌드 분석. 단어의 이미지 트렌드 변화 분석. 상품 출시 후 여론과 이미지를 키워드 수집으로 분석. topic modeling ,문서 군집화 기술을 사용.  WWW CIKM ICWSM KDD학회

정보검색: 검색 기술을 연구. 세부 항목으로 추천시스템. 스트리밍 서비스에서 비슷한 노래 자동 추천. 유투브 영상 자동 추천 등. 

우리 과정은 자연어처리를 주로 다룸. 

발달과정

임베딩: 단어를 벡터로 변환. 문장은 벡터 sequence으로 바꾼다. seq에 특화된 rnn이 자리 잡음. LSTM, 더 빠른 GRU가 사용되다가. 구글의 트랜스포머의 self attention으로 대체. 요즘은 트랜스포머가 기준. 원래 트랜스포머는 기계번역에 사용하기 위해 만듦. 이전에는 rule based. 전문가가 ... 언어 쌍 별로 문법 고려.

트랜스포머. 전문가 필요없음. seq data만 준다. 이제는 시계열 분석, 이미지에 모두 사용 됨.

과거에는 task마다 다른 architecture. 이제는 트랜스포머 범용 모델을 사용. transfer model으로 모든 task에 사용한다. 전문가 없이 할 수 있다고 해서 자가학습이러고 부름. 

기술적 트렌트: 자가지도학습을 위해서는 리소스와 데이터가 아주 많이 필요하다. gpt 학습 전기비용만 수십억이라고 함. 그래서 대기업들이 주도


bag of words
가장 간단한 벡터 전환 방법
텍스트 데이터에서 단어장 만들듯 사전 만드는 것. unique한 단어들만 담는다(그래서 bag of words). 그리고 one hot vector 만든다. 8개가 있다면... 각 단어들끼리의 거리는 root 2이고 모든 단어들간의 코사인 유사도는 0이다. 

이렇게 만든 vocab으로 단어들들 다 더해서 각 문장을 벡터로 만들 수 있음. 빈도 수별로 값이 커짐. 

Bag of words으로 만드는 document classifier. 문서의 주제를 분류한다. 일반적인 naive bayse classifier과 동일. 


word embedding
자연어가 단어의 sequence일 떄 단어를 벡터로 표현. 이 자체가 ML/DL. 학습 데이터를  텍스트 데이터로 준다. 벡터 차원을 하이퍼 파라미터. 그러면 그 차원에 맞게 단어를 임베딩 시킨다. 

아이디어: 비슷한 단어는 비슷한 좌표에 위치한 벡터. 벡터 만으로 유사도를 파악할 수 있음. 이걸 해두면 자연어 처리 DL할 떄 성능이 많이 오른다. love와 like은 비슷한 벡터 값. 이 벡터들을 사용한 값들도 비슷한 값. 같은 class으로 묶던지...

word2vec
한 단어가 주변 단어들의 의미들에서 파악 가능. 주변 단어들의 확률 분포를 통해 예측. cat을 예측하고 싶다? 주변 단들. meow. pet. 등의 단어가 높은 확률을 가진다는 것을 word2vec이 학습한다. 

과정
Sentence
tokenize
make vocab and one hot vector.
use sliding window to set context. 


# word2vec - skipgram
word2vec에 두가지 종류. 성능이 좋아서 대세인 skip gram에 대해 알아 봄.

목적: 단어를 벡터로 임베딩 시킬 떄 유사한 단어가 비슷한 위치에 있는 벡터가 되도록 임베딩 시킨다.
가정: 의미가 유사한 단어들은 주변 단어들의 확률 분포도 비슷할 것이다.

fake training: 단어를 임베딩 시키기 위해서 모델에게 주는 task. 임베딩 하고 싶은 단어의 의미는 주변 단어들에게 의존적일 것이다. 따라서 주변 단어들을 사용하여 임베딩 특징을 학습한다.

(임베딩할 목표인) 중심 단어 주변의 window size만큼 단어들을 추출. 각 단어를 one-hot vector으로 만든다. 중심 단어의 one hot vector을 input으로 넣는다. 하나의 중심 단어와 여러 주변 단어들의 조합이 각각 training example이 된다. (결과적으로 모든 단어에 대해서 이것을 반복하면 1epoch이 됨.)

![skipgram1](/assets/src/skipgram/skipgram1.png)

학습할 파라미터: 2개의 weight. 1개의 weight은 input에서 hidden layer으로 linear transformation. 다른 1개는 hidden layer에서 output layer으로 linear transformation. 

input -> weight1 -> hidden layer -> weight -> (softmax) -> output layer

여기서 input vector는 one hot vector이다. 그래서 input * weight1의 결과는 weight의 1 row가 그냥 뽑아져 나오는 것과 동일. 이 값이 hidden layer의 값. 따라서 hidden layer = weight1의 1개 row. weight1이 결과적으로 만들어지는 각 단어의 embedding matrix이다. weight의 1개 row가 각 단어의 embedding vector. 따라서 embedding vector의 차원은 나타내고자 하는 feature의 차원이다. 입력 벡터의 차원은 one hot vector을 구성하는 vocab의 크기와 동일하다. 따라서 weight1의 차원은 vocab_size * num_feature이다. hidden layer은 wieght1의 1개 row이기 때문에 차원 수는 num_feature이다. embedding matrix의 각 row는 각 중심 단어의 embedding vector이다.

![skipgram1](/assets/src/skipgram/skipgram2.png)

학습의 대상: 중심 단어를 집어넣었을 떄 지금 목표로 하는 특정 주변 단어가 나와야 한다. 따라서 나올 수 있는 모든 가능한 단어들 중에서 p(목표 주변 단어|input으로 들어간 중심단어)의 확률이 가장 높아야 함. 

현재 hidden layer의 차원 수는 num_feature이다. 나올 수 있는 모든 가능한 단어들은 vocab의 단어들. 이 모든 단어들의 확률이기 때문에 softmax을 사용하고, 따라서 output layer의 차원은 vocab_size이다. 이 중간을 잇는 weight2. weight2의 차원은 num_feature * vocab_size이다. 여기에 softmax. 그러면 최종 output은 입력된 중심 단어에서 나올 수 있는 vocab의 확률이다. 이 확률들과 목표한 주변 단어의 cross entropy을 구함. 목표 주변 단어 역시 one hot vector. 따라서 log likelihood에서 틀린 단어들의 확률은 다 0이 되고, 맞는 단어의 확률과 1의 차이만 계산 됨. 이 차이를 토대로 gradient descent하면... weight1과 weight2을 학습한다. weight1이 우리가 원하는 embedding matrix을 얻을 수 있다. 한 단어는 한 epoch 당 window size * 2만큼 학습 가능.  




## formalize
단어 사이즈(vocab size) $V$, feature size $M$, $i$번째 단어가 중심 단어. i의 window size에 속하는 j번쨰 단어가 현재 목표 주변 단어.

* input $1 \times V$
* weight1 $V \times M$
* hidden layer $1 \times M$
* weight2 $M \times V$
* outpult $1 \times V$

$i$th output node: $\frac{\exp H \cdot W_{2i}}{\sum_i \exp H \cdot W_{2i}} = p(j|i) = \hat y$

여기서 $W_{2i}$는 $W_2$ 행렬의 $i$ 번째 row이다.

loss 함수 CE : $y_i\log \hat y_i$

여기까지가 1개의 중심 단어에서 1개의 주변 단어를 feedforward하는 과정이다.

실제 코드구현은... 배치로 넣을 것이다! 

함께 들어가는 배치들은... 일단 1개의 중심단어에서 주변단어들이 모두 같이 들어간다. 그리고 T개의 중심단어들이 같이 들어감. 

그래서 최종적으로 논문에서와 같이 
![skipgram1](/assets/src/skipgram/skipgram4.png)



## pytorch
x에 1개, y에 window 1개. x에 1개, 다른 window 1개 이렇게 1개의 x에 대해서 다른 window을 하나씩 다 집어넣는다. get item에서 idx으로 1개 빼면 x 1개, window 1개 씩 빠진다. 1개의 for 반복에서 windiw*2개의 index을 생성해서 잡어넣는 중
```
class SkipGramDataset(Dataset):
  def __init__(self, train_tokenized, window_size=2):
    self.x = []
    self.y = []

    for tokens in tqdm(train_tokenized):
      token_ids = [w2i[token] for token in tokens]
      for i, id in enumerate(token_ids):
        if i-window_size >= 0 and i+window_size < len(token_ids):
          self.y += (token_ids[i-window_size:i] + token_ids[i+1:i+window_size+1])
          self.x += [id] * 2 * window_size

    self.x = torch.LongTensor(self.x)  # (전체 데이터 개수)
    self.y = torch.LongTensor(self.y)  # (전체 데이터 개수)

  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]
```

```
skipgram = SkipGram(vocab_size=len(w2i), dim=256)

batch_size=4
learning_rate = 5e-4
num_epochs = 5

skipgram_loader = DataLoader(skipgram_set, batch_size=batch_size)
```
미니 배치 크기 4으로 줬음. idx 1개에 x1개 y1개 나옴. 그래서 x4개 y4개씩 나올 것. 

이게 학습할 떄에는
```
class SkipGram(nn.Module):
  def __init__(self, vocab_size, dim):
    super(SkipGram, self).__init__()
    self.embedding = nn.Embedding(vocab_size, dim, sparse=True)
    self.linear = nn.Linear(dim, vocab_size)

  # B: batch size, W: window size, d_w: word embedding size, V: vocab size
  def forward(self, x): # x: (B)
    embeddings = self.embedding(x)  # (B, d_w)
    output = self.linear(embeddings)  # (B, V)
    return output
```

파이토치 nn embeding 함수 예시를 보면

[1,2,3,4]가 들어가면 (4, hidden_dim)이 나온다. 그리고 one hot vector으로 안바꾸고 넣어도 안에서 바꿔서 계산하는 듯. 원핫벡터이기 때문에 1개가 들어가면 1, dim이 나온다. n개가 들어가면 one hot vector n개라서 n, dim의 hiddden output이 나온다. 그래서 cbow는 window*2개가 들어가서 win*2, dim개가 나옴. skip gram은 1개가 들어가서 1개가 나옴. 그런데 이 기준에서 배치 사이즈대로 들어가니까, skipgram은 (B, dim)가 hidden output으로 나옴. 그리고 linear(d_w, V) 거쳐서 (1,d_w)가 (1, V)개가 된다. 

```
skipgram.train()
skipgram = skipgram.to(device)
optim = torch.optim.SGD(skipgram.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

for e in range(1, num_epochs+1):
  print("#" * 50)
  print(f"Epoch: {e}")
  for batch in tqdm(skipgram_loader):
    x, y = batch
    x, y = x.to(device), y.to(device) # (B, W), (B)
    output = skipgram(x)  # (B, V)

    optim.zero_grad()
    loss = loss_function(output, y)
    loss.backward()
    optim.step()
  
    print(f"Train loss: {loss.item()}")
print("Finished.")
```

loss함수로 nn.CrossEntropy을 쓰는데 토치의 CE에는 softmax가 포함되어 있음. y_true인 실수 1개가 label으로 전달. output으로는 class 수 만한 probability vector 전달. y_true에 해당하는 prob만 살아서 CE 계산. 그런데 4개 배치가 같이 들어감. 그리고 같이 들어간 배치는 argument으로 옵션 주지 않으면 평균내서 반환 함. 


$$

$$


![skipgram1](/assets/src/skipgram/skipgram3.png)

https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html

http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/30/word2vec/

https://arxiv.org/pdf/1310.4546.pdf

https://arxiv.org/pdf/1301.3781.pdf