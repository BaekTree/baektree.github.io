---
title: "Trainer_API_QA_task_Log"
last_modified_at: 2021-10-15T16:20:02-05:00
categories:
  - nlp
  - deep-learning
tags:
  - Huggingface
  - Trainer
  - issue
  - project
---
## 문제점
huggingface에서 QA task example에서... 

https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/trainer_qa.py

`python3 python train.py --output_dir ./outputs/train_dataset --do_train --model_name_or_path klue/roberta-small --do_eval --overwrite_output_dir --evaluation_strategy steps --logging_steps 100 --eval_steps 100` 이렇게 실행을 시키면, `do_eval`과 함께 `strategy steps`, `--eval_steps 100` 으로 되어 있어서 100 스텝마다 eval metric이 찍여야 함. 그런데 wandb을 보면 train/EM, train/F1으로 찍힘...
![image](https://user-images.githubusercontent.com/45053329/137110321-eda99aa5-5aef-4c62-b5b2-58af7186b2a8.png)


# 현재 wandb에서 eval이 안찍히는 이슈와 Trainer이 무슨 상관? 

## 현재 상황 분석
* baseline 코드를 보면 그냥 Trainer을 안쓰고 따로 만든 trainer_qa.py에서 trainer을 상속해서 쓰고 있다. QA task 상 loss 계산 후에 후처리를 해줘야 metric을 측정할 수 있기 때문이다. trainer_qa.py는 huggingface의 example에서 가지고 온것 같다. 

https://github.com/huggingface/transformers/blob/d9c62047a8d75e18d2849d345ab3394875a712ef/examples/question-answering/trainer_qa.py#L37

## 문제 분석
EM과 F1이 wandb에서 찍히기는 하는데, train/EM, train/F1으로 찍히는 것이 문제. Trainer 내부에서 wandb에 log을 찍는 과정을 살펴봐야 한다.


## 알아봐야 하는 점
* 기본 trainer와 trainer_qa의 차이점은 무엇인가?
* 어떻게 log을 찍는가?

# 기본 trainer과 train_qa의 evaluate 함수 차이점

* metrics prefix
* eval_loop 대신에 predict_loop을 넣음
* metrics.update 유무

```
# 기본 train
def evaluate(self, ... ):
    output = eval_loop(  compute_metrics, metric_key_prefix=metric_key_prefix, ... )
    self.log(output.metrics)

# train qa
def evaluate(self, ...):
    output= predict_loop(..., ) # compute_metrics 안들어간다. QA 특성 상 후처리를 해줘야 하기 때문이다.
    eval_preds = post_processing(output.prediction, ... ) # 후처리 여기서 함.
    metrics = compute metrics(eval_preds) # 최종 결과를 여기에서 metric 계산
    self.log(metrics)
```

정리하자면... 기본 train은 eval_loop에 compute metrics을 같이 집어넣어서 결과에 merics이 포함되어 있다.그리고 바로 log.
train qa는 predict_loop에 compute metrics을 빼고 넣는다. QA task에 대한 post processing을 해야 preds가 나오기 때문이다. 이걸로 다시 compute metrics을 수행한 결과를 log한다.


### 유의점: 
1. eval_loop과 predict_loopl이 다르다.
* prediction loop에는 loss 계산이 없음.

2. eval_loop에서 metric_key_prefix=metric_key_prefix 을 같이 넣는다.
* metric_key_prefix 받아서 metrics의 key에 삽입한다. 
* 그러면 eval_loop의 나온 결과가 self.log으로 들어간다. prefix을 eval으로 넣으면 wanbd에 저장될 때 eval section에 로그가 찍히게 된다.

# 문제의 근본적 원인
* train qa에서는 prefix을 넣지 않아서 log을 찍을 때 그냥 train 정보와 eval 정보를 하나로 다 취급해 버린다!

# 해결 방법
```py
# train qa
def evaluate(self, ...):
    output= predict_loop(..., ) # compute_metrics 안들어간다. QA 특성 상 후처리를 해줘야 하기 때문이다.
    eval_preds = post_processing(output.prediction, ... ) # 후처리 여기서 함.
    metrics = compute metrics(eval_preds) # 최종 결과를 여기에서 metric 계산
    ############
    for key in list(metrics.keys()):
        if not key.startswith(f"{metric_key_prefix}_"):
            metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
    ############
    self.log(metrics)
```

# 추가 이슈

## eval/loss을 계산해내지 못한다는 문제가 있음.
train qa의 evaluate을 뜯어보면... predict_loop을 뜯어야 하고... predict_step을 뜯다 보면... evaluation set이 label을 받아오지 않아서 loss 계산을 하지 않는다는 것을 알 수 있음. 

train의 경우 train_step에서 compute_loss을 호출하고, compute_loss에서 loss을 계산 함. evaluation과의 차이점은... prepare_train_feature에서 "answer"의 "strat_index", "end_index"을 함께 가져간다는 차이임. 실제 loss의 계산은 huggingface의 `AutoModelForQuestionAnswering`에서 계산하는 것처럼 보임. compute_loss에서 이 모델에 그냥 input까지 다 때려놓고 return 값만 loss으로 받아옴.

따라서 우리가 해야 할 일은 prepare_validation_feataure에서 train과 같이 answer의 start_index, end_index만 넣어주면 됨. 

그렇게 하면... eval loss을 prediction_loop에서 계산해낸다. 그걸 받아서 on_log에다가 넣는 metric 변수에 넣어주면 된다.

```py
            eval_loss = output.metrics['eval_loss']
            metrics['loss'] = eval_loss
```