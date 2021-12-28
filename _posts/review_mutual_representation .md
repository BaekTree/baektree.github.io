<!-- ---
title: "review MUTUAL INFORMATION MAXIMIZATION PERSPECTIVE OF LANGUAGE REPRESENTATION LEARNING"
last_modified_at: 2021-12-27T16:20:02-05:00
categories:
  - deep-learning
  - paper-review
tags:
  - DeepLearning
  - NLP
  - review
  - paper
--- -->

# A MUTUAL INFORMATION MAXIMIZATION PERSPECTIVE OF LANGUAGE REPRESENTATION LEARNING


* 제목
왠지... 학습을 할때 information의 정보 교환이 상호작용할 것 같다? 마치 self attention과 같은... 메커니즘? 그렇게 해서 상호작용을 극대화한다는 점에서 maximization이라고 한게 아닐까?

* 만든 사람들: 딥마인드, CMU, Google Brain...

항상 탑논문들을 보면 좋은 회사 좋은 학교... 우리 나라의 기업들/학교들도 이렇게 되었으면 좋겠다...!

* abstract
>We show state-of-the-art word representation learning methods maximize an objective function that is a lower bound on the mutual information between different
parts of a word sequence (i.e., a sentence). 

SOTA 논문들의 단어 representation 방법들은 다른 문장들과의 상호 정보에서 목적함수를 극대화(lower bound에 도달시킨다는?)한다는 것을 보였다. 이렇게 말하니까 뭔가 제목만 보고 예상한것이랑 비슷한것 같기도?

>Our formulation provides an alternative
perspective that unifies classical word embedding models (e.g., Skip-gram) and
modern contextual embeddings (e.g., BERT, XLNet). 

요즘 유행한 BERT 같이 문맥을 고려한 embedding과 기존의 skip gram같은 임베딩을 동일하게 하는 새로운 방향성을 제시! 

이게 뭔가 요즘 다시 기존의 전통적인 방법들이 다시 유행한다는 흐름과 맥을 같이 하는건가?

>. In addition to enhancing our
theoretical understanding of these methods, our derivation leads to a principled
framework that can be used to construct new self-supervised tasks

이 새로운 방향성을 바탕으로 새로운 self supervised task(아마도 pretrianing)을 제시

>by drawing inspirations from related methods based on mutual information maximization that have been successful in computer vision...  between a
global sentence representation and n-grams in the sentence...

CV의 방법론을 차용해음. 위에 말한대로 기존의 skip gram의 방식과 BERT에서 쓰는 방식을 혼합!?

>Our analysis offers a
holistic view of representation learning methods to transfer knowledge and translate progress across multiple domains (e.g., natural language processing, computer
vision, audio processing).

음성, CV, NLP에 모두 적용가능한 방법론이다!


* Conc
GLUE와 SQUAD에서 실험해봤다는 내용 말고는 새로운 내용은 딱히 없음. 

* introduction
>maximize a lower
bound on the mutual information between different parts of a word sequence.

다른 단어/문장들 사이의 상호 정보의 lower bound을 maximize한다!