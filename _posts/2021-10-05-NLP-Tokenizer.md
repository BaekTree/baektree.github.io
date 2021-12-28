---
title: "1005-Implementing-NLP-tokenizer"
last_modified_at: 2021-08-18T16:20:02-05:00
categories:
  - nlp
  - deep-learning
tags:
  - Tokenizer
  - pytorch
  - Preprocessing
  - tokenizer
  - NLP
---



# Implementing Tokenizer


```

class Tokenizer():
    def __init__(tokenizer, max_lengh, is_padding):
        self.is_padding = is_padding
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.pad_tok = "PAD"

    def tokenize(text):
        tokens = self.tokenizer(text)
        if len(tokens) < max_length:
            tokens += [self.pad_tok] * max_length - len(tokens)
        return tokens

    def batch_tokenize(batch1, batch2 = None):
        batch1_tokens = []
        for b in batch1:
            batch1_tokens.append( tokenize(b) )
        if batch2:
            batch2_tokenes = []
            for b in batch2:
                batch2_tokenes.append( tokenize(b) )
            return batch1_tokens, batch2_tokens
        else:
            return batch1_tokens

```

# huggingface tokenizer

* 실제 사용할 때 
  * instance을 그냥 다시 call 한다.
  * `__call__` 함수를 사용해서 instance을 이름을 함수처럼 사용.
  * tokenize의 문서를 찾아보면 __call__이 안보인다. 도대체 어케 쓰는거지?!?
  * `__call__`은 `transformers.tokenization_utils_base.py`의 `class PreTrainedTokenizerBase(SpecialTokensMixin, PushToHubMixin)`에 구현되어 있다. huggingface의 다른 토크나이저들이 모두 얘를 상속해서 구현한다.
  * 그래서 __call__을 실행하면, 익숙한 argument 이름들이 보인다.

```
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
```

* 보면 return type이 BatchEncoding이다.
  * __call__하면서 들어온 intput이 배치이면 batch_encode_plus을 실행한 결과를 반환한다.
  * batch_encode_plus가 실질적으로 우리가 아는 토크나이저의 역할(index encoding, padding, truncate, ...)을 수행한다. 그래서 이 batch_encode_plus부터 일반 문서에서 찾아볼 수가 있는 것!

```
            return self.batch_encode_plus(
                batch_text_or_text_pairs=batch_text_or_text_pairs,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
                **kwargs,
```
* 입력이 배치가 아니면 그냥 `encode_plus`에 입력을 다 때려박고 이 결과를 반환한다.

반환하는 함수들은 사실 BatchEncoding class의 instance을 반환한다. 이거 그냥 별거 없음. 그냥 python의 collection의 UserDict 상속해서 만든거임 ㅋㅋ . 여기에 그냥 인코딩한 정보들을 dict 형태로 넣은거. 그래서 attention mask, type id, 정수 인코딩 등등의 정보를 넣은게 전부! 

반환하는 애들

```
["input_ids", "token_type_ids", "attention_mask"]
```

끗.

https://huggingface.co/transformers/_modules/transformers/tokenization_utils_base.html

