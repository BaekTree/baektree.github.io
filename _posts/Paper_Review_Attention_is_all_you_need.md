**# Paper Review Attention is all you need**

**# Iteration 1**

**## title:**

Attention is all you need

제목이 도발적임.

**## figure1:**

architecture. encoder part and decoder part으로 구성 됨.

**## fig2:**

attention의 두가지 종류: scaled dot product and multi head attention

**## abstract:**

dominant seqence model: encoder + decoder with RNN and CNN with attention mechanism. Introducing Transformer, based only sorely on attention machanisms, despenses with RNN and CNN. Only use attentions. => make parallelizable, less time to train, achieves outperforming scores even including ensembles.

**## conclusion:**

Transformer replaces RNN and CNN in encoder-decoder architecture with attention with multi head self attention mechanism.

turned out transformer trains significantly faster than RNN or CNN. Also achieve a new state of art score including ensemble methods.

plan to apply to other task. such as input and output modalities other then test and attention mechanisms. to handle audios and videos, images. Making generation less sequential.

**# Iteration 2**

**## introduction**

RNN, LSTM used as a state of art method for ssequence modelings and transduction problems(****language modelingLM**** and ****machine translationMT****) using encoder-decoder.

- RNN

This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit

- Attention

Attention mechanisms have become an integral part

allowing modeling of dependencies without regard to their distance in the input or output sequences

however, such attention mechanisms are used in conjunction with a recurrent network. attention 만들기 위해서/그리고 사용할 때 encoder에서 RNN/BiRNN등을 미리 사용해야 했고 이 단계에서 기존 RNN의 단점이 그대로 적용.

we propose the Transformer

relying entirely on an attention mechanism to draw global dependencies between input and output. encoder과 decoder 사이를 연결하는데 attention만 사용했다. RNN 안씀!

The Transformer allows for significantly more parallelization and can reach a new state of the art

**## figure**

fig1:

fig2

**# Iteration 3**

2. background

CNN으로 RNN의 seq을 줄이고 병렬적으로 계산하려는 시도. input and output position에 대한 시간 복잡도가 선형/log으로 줄임.

the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet

트랜스포머는 이 시간 복잡도를 상수로 줄임. 어떻게? attention-weighted position을 평균했음. multi head attention의 원리이다.

In the Transformer this is reduced to a constant number of operations,

the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention

동일 sequence의 다른 위치를 represent할 수 있는 self attention!

Self-attention, sometimes called intra-attention is an attention mechanism ****relating different positions**** of ****a single sequence**** in order to compute a representation of the sequence

간단한 task에서는 RNN이 성능이 여전히 더 좋다

End-to-end memory networks are based on a recurrent attention mechanism instead of sequence- aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [34].

3. architecture

stacked self-attention layer와 pointwise FC 레이어를 사용함. 인코더와 디코더 모두에 사용.

Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder

인코더 stack

6개의 identical layers 사용

1개의 레이어는 2개의 sub layers.

하나는 multi head self attention.

다른 하나가 point-wise FC net이다.

여기에 residual connection으로 두 각 sub layer 시작과 끝에 연결해서 sum. 이 결과를 layer normalization에 넣는다. 모델 전체에서 큰 layer의 input과 output에 적용되는 dimension $d_{model}=512$이다.

디코더 stack

6개의 identical alyers. three sublayers per each layer. 근데 여기서 인코더의 sub layer 모양에서 세번째 sub layer가 드러감. 세번째 것은... encoder에서 나온 값에 대한 multi head attention을 수행하는 역할을 함. 각 sub layer 마다 역시 residual connection 연결해서 layer normalization을 한다. 그리고 디코더 역할에 알맞게 output을 다음 위치의 input으로 넣는다. 그리고 디코더의 input에서 이전 시점의 input에만 multi head attention 하기 위해서 마스킹을 아직 나오지 않은 위치에 대해서 masking 적용. dimension $d_{model}=512$으로 유지하기 때문에 residual sum 그대로 가능!

3.2 attention

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.

The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

sequence의 토큰 j가 있을 떄, 이 j와 비슷한 단어들에 집중(attention)할 것이다. 의미가 유사한 단어에 집중해야 함. 가장 단순한 방법은 cosine similarity의 의미를 가진 내적이다. 내적 값은 유사한 정도를 나타낼 것이다.

$$

\mathbf{a} \cdot \mathbf{b}=\|\mathbf{a}\|\|\mathbf{b}\| \cos \theta\\

$$

유사할 수록 positive으로 값이 커짐. 반대일수록 negative으로 값이 작아짐. orthoginal하면 0이다. 유사도를 가중치로 두고 각 토큰의 value에 적용한다.

1개의 query을 다른 모든 key와 value에 적용해서 weighted sum을 구했다. 즉 input query에 대한 attention output이 된다.  value도 벡터이므로, 전체 weighted sum도 벡터가 된다.

$$

A(q, K, V)=\sum_{i} \underbrace{\frac{\exp \left(q \cdot k^{<i}\right)}{\sum_{j} \exp \left(q \cdot k^{<j>}\right)}} v^{<i>}

$$

이떄 모든 토큰을 concat해서 matrix으로 만들어서 한번에 계산하면 for statement 보다 효율적이다. 그리고 각 토큰들에 대해서 softmax을 취해서 확률화시키고 값을 만든다.

for each query, for each key, for each value, vectorizing that...

$$

\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V

$$

각 query 마다의 attention 값이 vector이므로, 모든 query에 대한 최종 attention output은 행렬이 됨.

QK은 $d_k$ 차원이고, V는 $d_v$ 차원이다.

![ROC from wiki](/assets/src/transformer/qkv.png)

Q row는 1개의 query

K col은 1개의 key.

1개의 쿼리를 여러 K에 대해 내적하면 각각의 유사도가 나온다. softmax는 row을 기준으로 확률화. row가 각 v에 대한 쿼리의 유사도 가중치이다.

우리가 구하고자 하는 것은 weighted sum이다. 근데 v 하나가 벡터임. weighted sum을 해도 vector가 나온다. 각 원소 자리에 value 마다 동일한 weighted가 적용되는 것.

V와 내적하면. 각 d_v에 대해 적용해야 weighted sum이 된다.

1개의 query에 대해서(row), 모든 key(col)와 하나씩 내적한 결과(row)가 모든 value(row)에 pointwise product? 유사도의 크기가 가중치가 되어서 각 value에 곱해진다. 자기 자신의 WV가 가장 클 것.

https://towardsdatascience.com/transformer-networks-a-mathematical-explanation-why-scaling-the-dot-products-leads-to-more-stable-414f87391500

softmax는 큰 값에 몰아준다. 그래서 scale이 커지면 큰 값만 1에 가까워진다. d_k가 커질수록 내적에 속하는 원소들이 많아지고 내적값이 커진다. 자기 자신의 qk값이 가장 클 것이니까 그것만 1에 가까워짐. 그러면

scaled dot product에서√dk으로 나누는 이유에 대해 논문에서는 softmax에 들어가는 input 값이 너무 크면 gradient가 0에 가까운 값이 되어서 학습이 잘 되지 않는다고 표현되어 있는데요! 궁금해서 찾아본 내용 공유합니다!

요약: softmax는 input의 scale을 반영하지 않아서, input 벡터 중에 큰 값이 포함되어 있으면 1에 가까운 값으로 몰빵을 준다. softmax의 gradient는 softmax 값을 ( s x (1 - s ) )와 같이) 사용하는데, 1에 가까운 값이 있으면 전체 gradient 값이 0이 되어버린다. 그래서 back propagation이 모두 0에 가깝게 되어 학습이 잘 되지 않음. softmax의 input 값을 어느 정도 작게 하기 위해서 vector dimension에 맞춰서 나눠서 값을 줄여준다!

"We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients 4. To counteract this effect, we scale the dot products by 1 /√dk"

https://towardsdatascience.com/transformer-networks-a-mathematical-explanation-why-scaling-the-dot-products-leads-to-more-stable-414f87391500

3.2.2 Multihead attention

이제 여러개의 multihead을 병렬적으로 쌓는다.

head 마다 각각 다른 Representation을 가진다.

직관적인 의미: 다른 관점으로 상황을 보는 것.

concat으로 각각의 representation을 모아서 linear layer에 통과시켜서 차원 수를 맞추면서 값을 통합시킨다.

3.2.3  application of attention in our Model

이지만 나는 그냥 전체 과정을 설명해봐야겠다.

인코더

- 문장이 토큰으로 토크나이징된다.
- 토큰들이 embedding 되어서 이제 1개의 토큰은 벡터이다.
- sin, cosin 으로 정해진 positional embedding과 더해진다.
- 토큰 sequence가 인코더에 들어간다.
- 각 마다 3개의 독립적인 linear layer에 들어간다. 각각 key, query, value가 되어서 나온다.
- qkv가 self attention block에 진입한다.
- 각 q가 전체 다른 k에 대해서 scaled dot product 수행되고 softmax으로 유사도의 상대적인 정도가 확률화된다.
- 확률=가중치에 대해 value으로 가중평균한다. 이것이 한 토큰의 query에 대한 가중 평균 value 값이다. 직관적으로 한 토큰과 유사한 단어들의 가중평균 정보를 가지고 있다.
- 이 self attention block 8개가 동시에 병렬적으로 수행된다.
- 각 토큰에 대한 유사도가 포함되어 있는 value 가중 평균 정보들이 concat 된다.
- linear layer을 거치면서 정보가 종합적으로 합쳐지고 차원도 다시 원래 차원으로 복귀(d_model = 512).
- 이 multihead attention 값이 다시 key, query, value가 되어서 다음 multi head attention block에 들어간다.
- 들어가면서 각각 linear layer을 거쳐서 d_k, d_k, d_v 차원으로 변환.
- 6번 반복한다.
- 1개의 토큰에 대해서 512 차원의 벡터가 나온다.
- 이게 max len 만큼 있는 것.
- 그리고 배치 마다 있는 것.
- 이제 디코더
- 수행하려는 디코더 task에 따른 데이터가 준비되어야 한다.
- 인코더와 마찬가지로 문장이 토큰으로 분리 된다.
- 그리고 각 토큰 마다 벡터로 임베딩 된다.
- 각 토큰은 인코더와 마찬가지로 3개의 독립적인 linear layer을 지나서 key, query, value가 된다.
- 각각의 차원은 d_k, d_k, d_v가 된다.
- qkv가 multi attention에 들어가고 인코더와 마찬가지로 self attention을 수행.
- 그런데 이때... 디코더의 특성: 차례 차례 predict 한다. 따라서 실제 inference할 때와 학습할 때가 동일하게 해야 잘 학습. 그래서 self attention을 할때 masking을 넣어서 각 토큰이 들어갈 때 자기 오른쪽의 토큰들을 self attention을 하지 못하게 해야 한다.

[http://nlp.seas.harvard.edu/2018/04/03/attention.html#encoder-and-decoder-stacks](http://nlp.seas.harvard.edu/2018/04/03/attention.html#encoder-and-decoder-stacks)

```python
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
```

인코더에서 학습이 끝나면 각 토큰에 대해 attention vector가 나온다. 이걸 key와 value으로 생각해서 디코더에게 전달.

디코더에서 self attention을 한 결과 attention vector가 query가 된다. encoder에서 나온 값들과 함께 attention

1. why self attention

2. variants

# transformer

- 인코더 디코더 모델에서 시작!
- no more sequential! parellel!!!
- attention + cnn
- parellel with cnn style

## key ideas:

- self attention.

* e.g. 5 words. attention in parellel.

- multiple version

# self attention

$$

A(q, K, V)=\sum_{i} \underbrace{\frac{\exp \left(q \cdot k^{<i}\right)}{\sum_{j} \exp \left(q \cdot k^{<j>}\right)}} v^{<i>}

$$

## 트랜스포머 설명 바이 설명 transformer

https://nlpinkorean.github.io/illustrated-transformer/

- 번역 문제에 적용

1. 문장을 한꺼번에 다 집어넣는다. 병렬적으로.

2. 각 단어 마다 미리 설정해둔(pretrain이든 뭐든) 임베딩을 사용해서 dense word embeding 벡터들로 만든다.

## 트랜스포머 설명 바이 설명 transformer

https://nlpinkorean.github.io/illustrated-transformer/

- 번역 문제에 적용

1. 문장을 한꺼번에 다 집어넣는다. 병렬적으로.

2. 각 단어 마다 미리 설정해둔(pretrain이든 뭐든) 임베딩을 사용해서 dense word embeding 벡터들로 만든다.

3. 각 단어들 마다 파라미터 벡터 $W^q. W^k, W^v$을 내적시켜서 $q,k,v$ 벡터를 만든다.

4. 각 단어의 q와 다른 모든 단어들의 k을 내적사켜서 단어들 사이에 의미의 유사도를 찾는다. (코사인 유사도?).

5. 상수로 나눠서 값을 줄인다.

6. 그리고 소프트맥스를 해서 유사도에 대한 가중치를 만든다.

7. 그리고 각 단어의 가중치와 v을 곱해서 모두 더하면... 각 단어의 q와 유사한 단어들과의 value 가중 평균 값을 얻는다!

8. 이 과정을 병렬적으로 하기 위해서 행렬로 만든다. vectorization!

- 이러면 self attention이 끝. 이게 1개의 head이다.
- 이것을 서로 다른 8개를 실행 = multihead.
- featuremap과 유사?
- 서로 다른 hidden representation을 만드는 것.
- 각 head는 각 단어와의 유사성을 다르게 찾는다.
- 생각해볼 질문: q,k,v을 왜? 어떻게 생각하게 되었을까? 디비와의 연관성은?
- decoder에 넣는다.
- mask 사용ㅎ서 미래 쿼리 막는다
- 맨 처음 학습에 의한 기본 출력? startToken이 입력으로 들어온다.
- 그 출력이 입력 쿼리로 온다.
- 인코더에서 받은 k,v 매트릭스와 multihead와 내적 weighted sum.
- 합쳐서 벡터 출력.
- normalize 하고 ff 거쳐서 softmax으로 단어 예측 하고 argmax으로 가장 큰 단어 vocab에서 추출.