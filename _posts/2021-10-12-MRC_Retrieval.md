---
title: "MRC_Retrieval_Sparse_Embedding"
last_modified_at: 2021-10-12T16:20:02-05:00
categories:
  - NLP
  - camp
  - MRC
  - Retrieval
  - blog
---

DB/웹에 있는 필요한 조각이 있는 문서를 찾아주는 것. 문서를 가져오는 시스템.

왜 이게 필요한가? MRC 시스템과 연결하면 open domain system을 만들 수 있기 때문이다! 질문에 관련된 답을 가진 문서를 찾아서 MRC 모델로 넘긴다. 그럼 MRC 모델이 QA을 해서 질문자에게 넘겨 줌. 2stage pipeline.

이 과정은 문서를 찾아주는 과정이다. 

어떻게 구현? sparse embedding으로 구현한다. 오 이게 RAG에서 말했던... dense embedding 해두면 거기에서 retrieveal한다는거에서 dense 대신에 sparse인건가...!? 문서를 벡터로 만들어 둔다. 질문이 들어오면 vector로 만든다. 같은 space에 있는 문서를 찾음. 그리고 sim score으로 질문과 문서를 계산해서 가장 sim 높은 문서들을 반환.

어떻게 sparse embedding? 코사인/ 혹은 다른 유사도 공식 혹은 거리를 통해서 유사도를 계산. 각도로 표시하냐 거리로 표시하냐의 차이인 듯?

Sparse Embedding
단어 Bag Of Word 형태이다. vocab이 있으면... vocab에서 있는 단어들 수 카운트 해서 표현. 벡터의 크기가 vocab의 크기와 동일하게 됨.

n-gram: n개를 2개의 단어로 보고 vocab을 구성. 그래서... 만약 bi-gram이라고 했을 때, 단어의 수가 10개 이면 bigram voacb의 크기는 10^2이 된다... n gram이면 10^n이 됨. 단어의 수가 10만개면 bigram만 되어도 10만^2이 됨... 그래서 bigram 까지만 쓴다. 근데 왜 쓰나? vocab의 수가 어엄청 나게 커지는데... Bag of n-grams can be more informative than bag of words because they capture more context around each word (i.e. “love this dress” is more informative than just “dress”). 
http://uc-r.github.io/creating-text-features#ngrams

2. 빈도수 기반
TF-IDF 같은 것

Sparse 특징
등장하는 unique한 단어가 많을 수록 vocab의 크기가 증가
N gram에서 N이 커질 수록 vocab 크기가 증가

2. term이 있는지 없는지 파악 쉬움

그냥 count 수 등으로 알 수 있으니까.
단점: 의미가 비슷한 단어는 찾이 힘들다. 완전히 카운트 기반이라서 다른 벡터라서...

## 3. TF IDF
단어의 등장 빈도, 단어가 제공하는 정보의 양. 자주 등장하는 단어들

TF(t, d): d문서에서 t단어의 글자 수/총 단어 수. 자주 등장한 단어의 중요도 증가
IDF(t): 모든 문서에서 등장하는 단어(조사 등)은 중요하지 않으므로 중요도 감소. DF(t)는 t 단어가 등장하는 문서의 수. N = 전체 문서 수. 

a와 the 같은 단어의 idf는 낮아짐. 

$$
T F (t, d) = \frac{n u m b e r - o f - t}{ n u m b e r  - o f - a l l - t o k e n s }\\

I D F(t)=\log \frac{N}{D F(t)}
$$



모든 문서에서 등장하는 단어의 중요도는 낮추고, 각 문서에서 자주 등장하는 특징 단어의 중요도는 높이자!

$T F (t, d)$는 문서 d에서 토큰 t에 대한 실수 값이다. 따라서... 한 문장을 tfidf에 넣으면... 중요한 단어라는 기준에 따라서 각 토큰 별 실수가 되는 벡터가 나옴. 

쿼리와 가장 유사한 문서 랭크를 만들어보자!
1. 문서와 질의를 토큰화
2. 기존 단어 사전에 없는 토큰들은 제외(Sparse Embedding. 단어를 제어 가능. 불용어 제거)
3. 질의의 토큰에 대해서 각 문서와 질의에서 tF-idf 계산. tf(t, d), idf(t).
4. 질의 tfidf 벡터와 다른 문서들의 tfidf 벡터의 내적=코사인 유사도 계산.

$$
\operatorname{Score}(D, Q)=\sum_{t e r m \in Q} \operatorname{TFIDF}(\text { term, } Q) * \text { TFIDF }(\text { term, } D)
$$

BM25
tfidf에 문서의 길이까지 고려해서 점수. 길이가 작은 문서에 높은 점수. 아직도 많이 사용한다. challenge을 할때 bm25을 사용해서 개선할 수 있따.

$$
\operatorname{Score}(D, Q)=\sum_{t e r m \in Q} I D F(\text { term }) \cdot \frac{\operatorname{TFIDF}(\text { term, } D) \cdot\left(k_{1}+1\right)}{\operatorname{TFIDF}(\text { term }, D)+k_{1} \cdot\left(1-b+b \cdot \frac{|D|}{a v g d l}\right)}
$$

## 코드
* sklearn의 tfidfVectorizer 사용
* `ngram_range = (1,2)`으로 해서 bi-gram까지 사용.
* fit으로 input으로 들어온 corpus에 대해서 unique 단어 계산하고, bi-gram으로 vocab 만들고 tf-idf 계산한다.
* 결과: 각 문서에 대해서 unique bi-gram 크기 만큼 중요도 기준으로 각 토큰에 실수 매핑한다. 결과적으로 각 문장이 벡터가 된다!
* 이 경우는 9606개의 문장 courpus가 있고 벡터의 크기는(unique bigram vocab) 1,272,768개임.

```py
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(tokenizer=tokenizer_func, ngram_range=(1,2))from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(tokenizer=tokenizer_func, ngram_range=(1,2))

vectorizer.fit(corpus)
sp_matrix = vectorizer.transform(corpus)

sp_matrix.shape
>>(9606, 1272768)

```

