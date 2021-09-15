# subword tokenize

The main advantage of a subword tokenizer is that it interpolates between word-based and character-based tokenization. Common words get a slot in the vocabulary, but the tokenizer can fall back to word pieces and individual characters for unknown words.

https://www.tensorflow.org/text/guide/subwords_tokenizer

For tokens not appearing in the original vocabulary, it is designed that they should be replaced with a special token [UNK], which stands for unknown token.

However, converting all unseen tokens into [UNK] will take away a lot of information from the input data. Hence, BERT makes use of a WordPiece algorithm that breaks a word into several subwords, such that commonly seen subwords can also be represented by the model.

https://albertauyeung.github.io/2020/06/19/bert-tokenization.html


## huggingface tokenizer
* segment embedding
```
encoded_input = tokenizer("How old are you?", "I'm 6 years old")
print(encoded_input)

{'input_ids': [101, 1731, 1385, 1132, 1128, 136, 102, 146, 112, 182, 127, 1201, 1385, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

https://huggingface.co/transformers/preprocessing.html

* 스페이스, 마침표, 언어에 따른 rule based tokenizer

그냥 흔히 아는 토크나이저. 이 방법의 문제: 그냥 단어 별로 쪼개게 되면... 영어의 특성상 접두사, 접미사 같은 것들마다 단어 1개씩 별개로 토큰이 된다. 단어 수가 많아짐. BERT의 경우 임베딩까지 학습에 포함된다. 임베딩 쪽에 파라미터가 엄청 많아진다. 실제 모델 부분보다 임베딩 쪽에 파라미터가 더 많아지기도 한다. 파라미터가 많아지면 메모리도 많이 먹고 시간 복잡도도 커진다. 그 메모리 줄여서 블럭을 더 쌓을 수는 없을까?

* character 기반 tokenizer
단어 별로 토큰을 자르면 의미를 너무 많이 잃어버린다. 그래서 학습이 힘들어진다. today의 t와 그냥 t는 의미 차이가 너무 많이 난다.

* subword
character 기반은 너무 잘게 쪼개서 학습이 어렵고, space 기반은 너무 커서 또 메모리와 학습 시간이 너무 크다. 

중간은 없을까? 해서 나온 것이 subword 토큰이다. 

자주 나오는 단어들과 가끔 나오는 단어들을 구분한다. 자주 나오는 단어들은 그 자체로 사전에 저장하고, 가끔 나오는 단어들은 어떻게든 쪼갠다. 

i.e.
```
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.tokenize("I have a new GPU!")

["i", "have", "a", "new", "gp", "##u", "!"]
```

자주 안나오는(혹은 토크나이저가 처음보는) GPU같은 단어는 Unit의 U와 GP을 잘라서 저장한다. 
```
from transformers import XLNetTokenizer
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
tokenizer.tokenize("Don't you love 🤗 Transformers? We sure do.")

["▁Don", "'", "t", "▁you", "▁love", "▁", "🤗", "▁", "Transform", "ers", "?", "▁We", "▁sure", "▁do", "."]

```

이 경우에도 마찬가지. 자주 안나오는 Transformer라는 단어를 Transform과 er으로 구분했다. er도 e 와 r이 아니라 er으로 나눠진 것도 어느정도 등장해서 이렇게 된 것이다.

* 원리: 도대체 어떤 알고리즘으로?
Byte-Pair Encoding, WordPiece 등의 알고리즘이 있음. 

대부분 본격적인 토크나이징 전에 그냥 단어 별로 쪼갠다. 그리고 카운드를 해둔다. 그리고 이제 본격적으로...

## Byte-Pair Encoding
1. 카운드 되어 있는 단어들이 준비되었다
2. 각 단어들을 char 단위로 쪼갠다
3. char과 char이 가장 흔하게 붙는 경우를 찾아서 합쳐서 두글자로 만든다.
4. 계속 반복해서 하이퍼 파라미터로 주어지는 사전 수가 될때까지 반복!

GPT, RoBERTa에서 사용

Byte-level BPE는 바이트 단위로 한다고 한다... GPT2에서 사용

## WordPiece
Byte Pair Encoding과 비슷하다. 차이점은... 빈도 수로 char과 char을 합치는 것이 아니라, training이 가장 최적화되는 char 조합을 찾는다. `maximizing the likelihood of the training data`이다. 그러면 모델 전체 학습 과정에서 이것도 학습되나?

BERT, DistilBERT, Electra에서 사용.

* 이것 말고도 Unigram이 있다고 함.




https://huggingface.co/transformers/tokenizer_summary.html
