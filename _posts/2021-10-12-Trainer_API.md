---
title: "Trainer_API"
last_modified_at: 2021-10-13T16:20:02-05:00
categories:
  - NLP
  - Huggingface
  - Trainer
---

# 사전 지식: 
## Trainer: 
native pytorch 코드가 아니라 간단하게 training argument, 사용할 metric 함수, dataset만 던져 넣으면 알아서 학습을 돌려준다.
그러면 내부에서 epoch 마다 돌리고, step 마다 돌리고, loss 계산하고, gradient 계산하고, evaluation 계산하고, 다 해준다 ^^ Customization을 하려면 Trainer을 상속하던지, 제공하는 API에 맞게 custom function, argument을 만들어서 집어넣어주면 된다.
```py
trainer = Trainer( 
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
trainer.tarin()
```

## Callbacks: 
기본적인 Trainer 기능에 추가 기능을 넣는 것이라고 생각하면 편함. e.g Ealry Stop Callback, TensorBoard Callback, Wandb Callback 등이 있음. 

```py
trainer = Trainer( 
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks = [ ... callbacks list ...],
        compute_metrics=compute_metrics,
    )
trainer.tarin()
```

## TrainerState
Trainer 내부에서 학습의 상태를 저장해두고 있는 object이다. TrainerState class에서 생성 됨. Callback과는 다르게 겉으로 드러나지 않고 Trainer 내부에서만 동작한다. trainer = Traner(...)으로 trainer obejct을 생성할 때 같이 생성 됨.

## TrainerControl
Trainer 내부에서 학습을 도는 중간 중간에 특정 조건에 따라 특정 작업을 해야 할때 Control object에 있는 property를 조건문으로 사용하여 제어한다. should_log? should_evaluete? should_stop? 등의 정보를 가지고 있음. trainer = Traner(...)으로 trainer obejct을 생성할 때 TrainerControl class로부터 생성 됨.

# Trainer API을 쓸때 사용하는 Class들
## Trainer Class
* 실제 epoch 돌고 학습하는 모든 과정을 가진 클래스. 여기에서 밑의 다른 Class들을 다 import 해서 객체를 만든다. 저것들 활용해서 학습 과정을 제어함.
## TrainerControl Class
* 위 설명 참고
## TrainerCallback Class
* 위에서는 필요한 추가기능 Callback을 만들어서 집어넣어준다고 했었음. 그런데 Callback을 만들때도 Trainer가 받아들일 수 있도록 정해진 interface에 따라서 만들어야 함. 그 기준이 되는 interface class가 TrainerCallback Class이다. 뜯어보면 구현이 안되어 있음.
## TranierState Class
* 위 설명 참고
## CallbackHandler Class
* Trainer가 받아들이는 Callback들을 init 하고, Trainer가 필요한 때에 불러서 사용할 수 있도록 대 객체로 만들어서 저장하고 있는 object을 만들어내는 class.
Trainer Class 내부 구조
* 의사 코드 형태로 어떻게 돌아가는지 파악해보기
* 주석만 읽어도 돼요!
* 위에 있는 callback handler, train control, train state, callback의 의미를 상기하면서 읽으면 조음.

```py
class Trainer():
    def __init__(self, model, train_arg, compute_metrics, callbacks, train_set, eval_set, ...)
        # init하면 argument으로 받은 값들을 calss property으로 저장함. 
         self.model = model
         # 추가 기능 callbacks들은 CallbackHandler에 저장해두고, CallbackHandler 만을 가지고 있는다. 특정 조건과 시점에 callback의 추가기능을 사용할 것임.
         self.callback_handler = CallbackHandler(callbacks, ... )
         self.args = train_arg

        # 현재 학습 상태를 기록해둔다. 
        self.state = TrainerState()
        # state을 살펴보면서 제어를 해야 할 타이밍인지 여부를 기록해둔다.
        self.control = TrainerControl()
    
    def get_train_dataloader(self):
        ...
        train_dataset = self.train_dataset
        ...

        return DataLoader(train_dataset, self.arg의 다양한 조건들, ... )
    
    def get_eval_dataloader(self):
        ...
        eval_dataset = self.eval_dataset
        ...

        return DataLoader(eval_dataset, self.arg의 다양한 조건들, ... )

    def get_test_dataloader(self):
        ...
        test_dataset = self.test_dataset
        ...

        return DataLoader(test_dataset, self.arg의 다양한 조건들, ... )

    def train(self, resume_from_checkpoint, ...):
        # 만약 checkpoint가 있으면 그걸로 model을 대체함.
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint

        ...

        if resume_from_checkpoint is not None:
            ...
            # init에서 self.model이 기존 checkpoint model으로 대체 됨.
            self._load_state_dict_in_model(state_dict) 
            
        train_dataloader = self.get_train_dataloader()

        # 학습에 필요한 변수들, grad 초기화
        self.state.epoch = 0
        tr_loss = torch.tensor(0.0).to(args.device)
        model.zero_grad()

        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloade

        # 학습 직전에 필요한 추가기능을 여기서 실행시킴
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # train arg에서 받은 arg으로 몇번 학습할지 설정
        num_train_epochs = math.ceil(args.num_train_epochs)
        epochs_trained = 0 # if resumed, change to the last epoch

        # epoch 반복
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader

            # 각 미니배치 반복.
            # 1 step 학습 = 1개의 미니배치 단위 학습(32, 64, ...)
            for step, inputs in enumerate(epoch_iterator):
              tr_loss_step = self.training_step(model, inputs)
              tr_loss += tr_loss_step
              model.zero_grad()

              # 현재 학습 시점 업데이트
              self.state.epoch = epoch + (step + 1) / steps_in_epoch

              # 1 스텝 끝났을 때 필요한 추가기능 실행
              self.control = self.callback_handler.on_step_end(args, self.state, self.control)

              # 현재 스텝이 evaluatoin과 log을 해야하는 상황이면 eval 하고 log 수행
              # 이 이슈 해결하려면 여기를 봐야 함
              self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            # 1 epoch 끝났을 때 필요한 추가기능 실행
            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            # 현재 epoch이 evaluatoin과 log을 해야하는 상황이면 eval 하고 log 수행
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

        # 학습 종료 후 결과 저장해서 log 찍음.
        metrics = {}
        metrics["train_loss"] = train_loss
        self.log(metrics)

        # 학습 종료 후 해야 하는 추가 기능 실행
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        
    # 현재 상황이 evaluatoin과 log을 해야하는 상황이면 eval 하고 log 수행
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        # 지금 상황이 log 찍어야 하는 상황인가?
        # train_arguemnt에서 --log_step = 100 등에서 조건 만족되면 True
        if self.control.should_log:
            logs: Dict[str, float] = {}
        
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)

            # 실제 log 찍는 함수. console, wandb 등이랑 모두 연결
            self.log(logs)

        metrics = None
        # 지금 상황이 evaluation 해야 하는 상황인가?
        # train_arguemnt에서 --eval_step = 100 등에서 조건 만족되면 True
        if self.control.should_evaluate:
            # 이 이슈 해결하려면 여기를 봐야 함 
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
    
    # 실제 log 찍는 함수. console, wandb 등이랑 모두 연결
    def log(self, logs: Dict[str, float]) -> None:
        # log 찍어야 하는 추가 기능을 실행.
        # wandb을 켰다면 callback hanlder 내부에서 연결되어 있음.
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        # eval_loop이 evaluation의 Trainer라고 보면 됨. eval_data 넣어서 loss 다 구해서 반환.
        # 이 이슈 해결하려면 여기를 봐야 함 
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # 실제 log 찍는 함수. console, wandb 등이랑 모두 연결
        self.log(output.metrics)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        ...
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))

       # 이슈 해결의 핵심. prefix에 eval을 달아준다.
        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
```

# 어떻게 log을 찍는가?


```py
# train.py
    def evaluate(self, ... ):
        output = eval_loop(  compute_metrics, metric_key_prefix=metric_key_prefix, ... )
        self.log(output.metrics)

    # 실제 log 찍는 함수. console, wandb 등이랑 모두 연결
    def log(self, logs: Dict[str, float]) -> None:
        # log 찍어야 하는 추가 기능을 실행.
        # wandb을 켰다면 callback hanlder 내부에서 연결되어 있음.
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, log
```

## 추가 기능 callback의 내부 구조
를 알아야 함 ㅜㅜ 

trainer을 init 할때 callback list을 받음.
받아서 callback_handler에 때려넣으면 callback_handler instance가 각 callback instance을 가지고 있음. 
```py
class Trainer():
    def __init__(self, model, train_arg, compute_metrics, callbacks, train_set, eval_set, ...)
         # 추가 기능 callbacks들은 CallbackHandler에 저장해두고, CallbackHandler 만을 가지고 있는다. 특정 조건과 시점에 callback의 추가기능을 사용할 것임.
         self.callback_handler = CallbackHandler(callbacks, ... )
````

여기서 wandb가 실행되고 있으면 wandb Callback도 같이 생성되어서 callback_handler에 저장되어 있다.

그리고 callback들이  실행되는 시점을 보면 `on_` 으로 시작하는 prefix 함수들을 가지고 있다. 모든 Callback class는 TrainerCallback interface을 상속해서 구현하고 있음. 그리고 callback_handler 역시 이 함수들을 가지고 있다. 따라서 callback_handler가 `on_` 함수를 실행하면 해당 `on_`을 가지고 있는 callback들이 모두 실행 됨.


e.g. `self.callback_handler.on_train_begin(...)` 을 실행하면 callback_handler에 있는 모든 callback instance들이 on_train_begin을 실행 함.

```py
class TrainerCallback:
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of training.
        """
        pass

class CallbackHandler(TrainerCallback):

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_training_stop = False
        return self.call_event("on_train_begin", args, state, control)

    def call_event(self, event, args, state, control, **kwargs):
        for callback in self.callbacks:
            result = getattr(callback, event)(
                args,
                state,
                control,
                model=self.model,
                tokenizer=self.tokenizer,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                **kwargs,
            )
            # A Callback can skip the return of `control` if it doesn't change it.
            if result is not None:
                control = result
        return control


class TensorBoardCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):

class WandbCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
```

## on_log
on_log도 마찬가지이다. 

```py
# train.py
    def evaluate(self, ... ):
        output = eval_loop(  compute_metrics, metric_key_prefix=metric_key_prefix, ... )
        self.log(output.metrics)

    # 실제 log 찍는 함수. console, wandb 등이랑 모두 연결
    def log(self, logs: Dict[str, float]) -> None:
        # log 찍어야 하는 추가 기능을 실행.
        # wandb을 켰다면 callback hanlder 내부에서 연결되어 있음.
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, log
```
위 함수에서 log을 실행하고 wandb가 켜져 있으면 wandb callback이 실행 됨.
```py

def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d


class WandbCallback(TrainerCallback):
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            logs = rewrite_logs(logs)
            self._wandb.log({**logs, "train/global_step": state.global_step})
```

입력으로 들어오는 log dictionary에 prefix으로 eval이 찍혀 있으면 eval으로 구분해서 eval section에 로그 찍음.

따라서

# 해결 방법
wandb log에 들어가는 `self.log(metrics)`을 실행할 때 `metrics` dictionary에 eval prefix을 추가하면 된다.