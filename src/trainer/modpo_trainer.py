from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from src.trainer.dpo_trainer import DPOTrainer, DPODataMapFunc, DPODataCollatorWithPadding
from src.utils.reward import RewardWrapperList, RewardWrapperInput


@dataclass
class MODPODataMapFunc(DPODataMapFunc):
    def __call__(self, examples):
        """
        Additionally keep untokenized prompts (`raw_prompt`) and responses (`chosen`, `rejected`)
        in the batch for easy adaptation for customized margin reward models (`src.utils.RewardWrapperBase`).

        For example, margin reward models can be an external API than depends on raw texts.
        """
        new_examples = super().__call__(examples)
        new_examples["raw_prompt"] = examples["raw_prompt"]
        new_examples["chosen"] = examples["chosen"]
        new_examples["rejected"] = examples["rejected"]
        return new_examples


@dataclass
class MODPODataCollatorWithPadding(DPODataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]], generate: Optional[bool] = False) -> Dict[str, Any]:
        batch = super().__call__(features, generate)
        if not generate:
            batch["raw_prompt"] = [feature["raw_prompt"] for feature in features]*2
            batch["response"] = [feature["chosen"] for feature in features] + [feature["rejected"] for feature in features]
        return batch


class MODPOTrainer(DPOTrainer):
    """
    The MODPOTrainer is a light-weight extension of DPOTrainer that supports training with 
    multiple margin reward models for multi-objective alignment. 

    Please use `set_wrapped_margin_reward_model_list` to set your customized margin reward models
    (`wrapped_margin_reward_model_list`) and the weights for each objective (`w`).
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        beta: float = 0.1,
        loss_type: Literal["sigmoid", "hinge"] = "sigmoid",
        args: TrainingArguments = None,
        tokenize_map_func: Optional[Callable] = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[Dict] = None,
        disable_dropout: bool = True,
        max_length: Optional[int] = 1024,
        num_proc: Optional[int] = 4,
        generate_during_eval: bool = True,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
    ):

        if tokenize_map_func is None:
            tokenize_map_func = MODPODataMapFunc(tokenizer)

        if data_collator is None:
            data_collator = MODPODataCollatorWithPadding(tokenizer)

        super().__init__(
            model=model,
            ref_model=ref_model,
            beta=beta,
            loss_type=loss_type,
            args=args,
            tokenize_map_func=tokenize_map_func,
            data_collator=data_collator,
            label_pad_token_id=label_pad_token_id,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            disable_dropout=disable_dropout,
            max_length=max_length,
            num_proc=num_proc, 
            generate_during_eval=generate_during_eval,
            compute_metrics=compute_metrics,
        )
    
    def set_wrapped_margin_reward_model_list(
        self, 
        wrapped_margin_reward_model_list: RewardWrapperList, 
        w: List[float],
        prepare: Optional[bool] = True,
    ):
        """
        Set margin reward models.

        Args:
            wrapped_margin_reward_model_list (`src.utils.RewardWrapperList`):
                A list of reward model to act as margin in `modpo_loss`. 
            w (`List[float]`):
                A list of weights for each objective. Note that w[0] indicates the weight for 
                the preference that we are currently training on and w[1:] indicate the weights for
                the margin reward models in `wrapped_margin_reward_model_list`.
            prepare (`bool`):
                Whether or not we need to `self.accelerator.prepare_model` the margin reward models for advanced distributed training. 
                If these margin reward models are part of the self.model (e.g, lora weights), they will have been prepared
                in `__init__` and we would recommend `prepare=False` to avoid unnecessary model weights copies.
                See `scripts/modpo/beavertails/modpo.py` for a complete example.
        """
        if prepare:
            def prepare(wrapped_reward_model):
                if hasattr(wrapped_reward_model, "model"):
                    wrapped_reward_model.model = self.accelerator.prepare_model(
                        wrapped_reward_model.model, evaluation_mode=True)
                return wrapped_reward_model
            wrapped_margin_reward_model_list = wrapped_margin_reward_model_list.map(prepare)
        self.wrapped_margin_reward_model_list = wrapped_margin_reward_model_list 
        self.w = torch.tensor(w).to(self.accelerator.device)
        assert len(self.wrapped_margin_reward_model_list) == len(self.w) - 1

    def modpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        chosen_margin_reward: torch.FloatTensor,
        rejected_margin_reward: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        chosen_rewards   = (1/self.w[0])*(self.beta * (policy_chosen_logps   - reference_chosen_logps)   - chosen_margin_reward   @ self.w[1:])
        rejected_rewards = (1/self.w[0])*(self.beta * (policy_rejected_logps - reference_rejected_logps) - rejected_margin_reward @ self.w[1:])

        logits = chosen_rewards - rejected_rewards
        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(logits)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - logits)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']")

        return losses, chosen_rewards.detach(), rejected_rewards.detach()

    def dpo_loss(self, *args, **kwargs):
        """Disable the `dpo_loss` inherited from the DPOTrainer"""
        raise NotImplementedError

    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            _,
            _,
        ) = self.forward(model, batch)
        with torch.no_grad():
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
            ) = self.forward(self.ref_model, batch)

        margin_reward_list = self.wrapped_margin_reward_model_list(
            RewardWrapperInput(raw_prompt=batch["raw_prompt"], response=batch["response"]))
        margin_rewards = torch.stack(margin_reward_list, dim=-1).to(
            policy_chosen_logps.dtype).to(self.accelerator.device) # (B*2, n-1)
        chosen_margin_rewards, rejected_margin_rewards = margin_rewards.chunk(2) # (B, n-1)
         
        losses, chosen_rewards, rejected_rewards = self.modpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            chosen_margin_rewards,
            rejected_margin_rewards,
        )

        accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu()
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu()
        metrics[f"{prefix}logps/margins"] = (policy_chosen_logps - policy_rejected_logps).detach().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu()
        if train_eval == "train":
            metrics[f"{prefix}accuracy"] = accuracies.detach().cpu()

        return losses.mean(), metrics
