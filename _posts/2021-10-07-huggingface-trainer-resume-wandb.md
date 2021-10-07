---
title: "2021-huggingface-trainer-resume-wandb"
last_modified_at: 2021-08-18T16:20:02-05:00
categories:
  - NLP
  - Tokenizer
  - Preprocessing
---

huggingface의 trainer api을 쓰면서 wandb을 쓸때.

용량 없거나 이런저런 이유로 도중에 중단 됨.

resume 하는 법.

1. run id을 받는다. run 폴더에서 가장 나중에 실행된 id을 받을 수도 있고 wandb 사이트에서 overview에 가면 id을 받아올 수 있음.
2. 다시 이어서 실행할 checkpoint 경로를 받아옴.

```

wandb.init(id="3fdn2tkl", resume="allow")
# wandb.init(project='klue-RE', name=cfg['wandb']['name'],tags=cfg['wandb']['tags'], group=cfg['wandb']['group'], entity='boostcamp-nlp-06')
    
```

기존의 wandb init 코드의 argument 대신에 id을 넣는다 그리고 allow 혹은 must을 넣음. must으로 넣으면 id가 없을 때 에러 내면서 멈춤. allow는 해당 id 없으면 처음부터 시작.

그리고 기존대로 training api 채우고 실행할 때
```
    trainer.train("/opt/ml/remote/results/roberta_large_stratified_using_MLM_1100_exp/checkpoint-1400")

```

기존에는 그냥 `trainer.train()` 이겠지만 argument으로 checkpoint가 있는 디렉토리 경로를 넣음.

그리고 `python3 train.py`으로 실행하면... 기존의 epoch들은 스킵하고 지나감. 

# 주의
간혹 id을 잘못넣었거나 checkpoint 경로를 잘못넣거나 다른 곳에 넣으면 에러 뜨면서 안돌아감.
 
 wandb init 되면서 wandb 디렉토리에 각 id 마다 폴더를 하나씩 만듬. 근데 resume 할때도 동일한 id의 폴더를 만듬. 실패하든 성공하든 만듬. 문제는... 공식 문서에서는 동일한 id가 있을 때는 unexpected result가 나온다고 함. 실패할 때는 알아서 폴더를 안만들면 참 조을텐데 ㅜㅜ 무튼 그래서 실패할 경우 wandb에서 해당 id에 대한 폴더를 삭제하는게 좋을 것 같다. 끗.