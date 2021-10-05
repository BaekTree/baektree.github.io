---
title: "1005-Implementing-NLP-tokenizer"
last_modified_at: 2021-08-18T16:20:02-05:00
categories:
  - NLP
  - Tokenizer
  - Preprocessing
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