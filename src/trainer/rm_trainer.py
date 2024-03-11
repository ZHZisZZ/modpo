from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import torch
import torch.nn as nn
import numpy as np
from datasets import Dataset
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import EvalPrediction
from trl.import_utils import is_peft_available
from trl.trainer.training_configs import RewardConfig
from trl.trainer.utils import PeftSavingCallback, RewardDataCollatorWithPadding, compute_accuracy

from src.utils import print_local_main, prepare_model_for_peft


if is_peft_available():
    from peft import PeftModel


@dataclass
class RewardDataMapFunc:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, examples):
        prompt_chosen   = [prompt + chosen   for prompt, chosen   in zip(examples["prompt"], examples["chosen"])]
        prompt_rejected = [prompt + rejected for prompt, rejected in zip(examples["prompt"], examples["rejected"])]
        chosen_sequence_tokens   = self.tokenizer(prompt_chosen)
        rejected_sequence_tokens = self.tokenizer(prompt_rejected)

        return {
            "prompt_chosen_input_ids":        chosen_sequence_tokens["input_ids"],
            "prompt_chosen_attention_mask":   chosen_sequence_tokens["attention_mask"],
            "prompt_rejected_input_ids":      rejected_sequence_tokens["input_ids"],
            "prompt_rejected_attention_mask": rejected_sequence_tokens["attention_mask"],
            **({"margin": examples["margin"]} if "margin" in examples else {}),
        }


@dataclass
class RewardDataCollatorWithPadding:
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        pad_to_multiple_of (`Optional[int]`, `optional`, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, `optional`, defaults to `"pt"`):
            The tensor type to use.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        batch = {
            "input_ids": ...,
            "attention_mask": ...,
            "return_loss": True,
            "margin": ..., (optional)
        }
        """
        new_features = []
        # check if we have a margin. If we do, we need to batch it as well
        margin = []
        if "margin" in features[0]:
            margin = [feature["margin"] for feature in features]
        for feature in features:
            new_features.append(
                {
                    "input_ids": feature["prompt_chosen_input_ids"],
                    "attention_mask": feature["prompt_chosen_attention_mask"],
                }
            )
        for feature in features:
            new_features.append(
                {
                    "input_ids": feature["prompt_rejected_input_ids"],
                    "attention_mask": feature["prompt_rejected_attention_mask"],
                }
            )
        batch = self.tokenizer.pad(
            new_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["return_loss"] = True,
        if margin:
            margin = torch.tensor(margin, dtype=torch.float)
            batch["margin"] = margin
        return batch


class RewardTrainer(Trainer):
    r"""
    The RewardTrainer can be used to train your custom Reward Model. It is a subclass of the
    `transformers.Trainer` class and inherits all of its attributes and methods. It is recommended to use
    an `AutoModelForSequenceClassification` as the reward model. The reward model should be trained on a dataset
    of paired examples, where each example is a tuple of two sequences. The reward model should be trained to
    predict which example in the pair is more relevant to the task at hand.

    The reward trainer expects a very specific format for the dataset. The dataset should contain two 4 entries at least
    if you don't use the default `RewardDataCollatorWithPadding` data collator. The entries should be named
    - `input_ids_chosen`
    - `attention_mask_chosen`
    - `input_ids_rejected`
    - `attention_mask_rejected`

    Optionally, you can also pass a `margin` entry to the dataset. This entry should contain the margin used to modulate the
    loss of the reward model as outlined in https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/.
    If you don't pass a margin, no margin will be used.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: Optional[RewardConfig] = None,
        tokenize_map_func: Optional[Callable] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[Dict] = None,
        max_length: Optional[int] = 1024,
        filter_too_long: Optional[bool] = True,
        num_proc: Optional[int] = 4,
    ):
        """
        Initialize RewardTrainer.

        Args:
            model (`transformers.PreTrainedModel`):
                The model to train, preferably an `AutoModelForSequenceClassification`.
            args (`RewardConfig`):
                The arguments to use for training.
            data_collator (`transformers.DataCollator`):
                The data collator to use for training. If None is specified, the default data collator (`RewardDataCollatorWithPadding`) will be used
                which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
            train_dataset (`datasets.Dataset`):
                The dataset to use for training.
            eval_dataset (`datasets.Dataset`):
                The dataset to use for evaluation.
            tokenizer (`transformers.PreTrainedTokenizerBase`):
                The tokenizer to use for training. This argument is required if you want to use the default data collator.
            model_init (`Callable[[], transformers.PreTrainedModel]`):
                The model initializer to use for training. If None is specified, the default model initializer will be used.
            compute_metrics (`Callable[[transformers.EvalPrediction], Dict]`, *optional* defaults to `compute_accuracy`):
                The metrics to use for evaluation. If no metrics are specified, the default metric (`compute_accuracy`) will be used.
            callbacks (`List[transformers.TrainerCallback]`):
                The callbacks to use for training.
            optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
                The optimizer and scheduler to use for training.
            preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
                The function to use to preprocess the logits before computing the metrics.
            peft_config (`Dict`, defaults to `None`):
                The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        """
        if not isinstance(model, PeftModel) and is_peft_available() and peft_config:
            model = prepare_model_for_peft(model, peft_config, args)

        if tokenize_map_func is None:
            tokenize_map_func = RewardDataMapFunc(tokenizer)

        if data_collator is None:
            data_collator = RewardDataCollatorWithPadding(tokenizer)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
            if getattr(tokenizer, "pad_token", None) is None:
                tokenizer.pad_token = tokenizer.eos_token

        def preprocess_dataset(dataset):
            # tokenize samples
            dataset  = dataset.map(
                tokenize_map_func, 
                batched=True, 
                num_proc=num_proc, 
                remove_columns=dataset.column_names
            )
            original_length = len(dataset)
            # filter samples that are too long
            if filter_too_long:
                dataset  = dataset.filter(
                    lambda x: len(x["prompt_chosen_input_ids"]) <= max_length and len(x["prompt_rejected_input_ids"]) <= max_length
                )
            else:
                dataset = dataset.map(
                    # truncate chosen and rejected
                    lambda sample: {k: v[:max_length] if ('chosen' in k or 'rejected' in k) else v for k, v in sample.items()},
                    num_proc=num_proc,
                )
            filtered_length = len(dataset)
            return dataset, filtered_length / original_length
        preprocess_here = False
        if "prompt_chosen_input_ids" not in train_dataset[0].keys():
            preprocess_here = True
            print_local_main("dataset preprocessing...")
            train_dataset, train_dataset_retain = preprocess_dataset(train_dataset)
            eval_dataset, eval_dataset_retain = preprocess_dataset(eval_dataset)
            print_local_main(f"train_dataset_retain: {train_dataset_retain}")
            print_local_main(f"eval_dataset_retain: {eval_dataset_retain}")

        if is_peft_available() and isinstance(model, PeftModel):
            if callbacks is None:
                callbacks = [PeftSavingCallback()]
            else:
                callbacks += [PeftSavingCallback()]

        if compute_metrics is None:
            compute_metrics = compute_accuracy

        self._stored_metrics = defaultdict(lambda: defaultdict(list))
            
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

        if preprocess_here:
            self.log({"eval_dataset_retain": train_dataset_retain, "dataset_retain": eval_dataset_retain})

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        rewards = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )[0]
        chosen_rewards, rejected_rewards = rewards.chunk(2)
        # calculate loss, optionally modulate with margin
        if "margin" in inputs:
            loss = -nn.functional.logsigmoid(chosen_rewards - rejected_rewards - inputs["margin"]).mean()
        else:
            loss = -nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()

        # force log the metrics
        if self.accelerator.is_main_process:
            accuracy = (chosen_rewards > rejected_rewards).float().detach().cpu()
            self.store_metrics({"accuracy": accuracy}, train_eval="train")

        if return_outputs:
            return loss, {
                "chosen_rewards": chosen_rewards,
                "rejected_rewards": rejected_rewards,
            }
        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, logits_dict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        loss = loss.detach()
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = nested_detach(logits)
        # Stack accepted against rejected, mean over logits
        # and softmax to get preferences between accepted and rejected to sum to 1
        logits = torch.stack(logits).mean(dim=2).softmax(dim=0).T

        labels = torch.zeros(logits.shape[0])
        labels = self._prepare_inputs(labels)

        return loss, logits, labels

    def store_metrics(self, metrics: Dict[str, np.ndarray], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value.mean())

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)


if __name__ == '__main__':
    from src.data.configs import DATASET_CONFIGS
    from transformers import LlamaTokenizer
    dataset   = DATASET_CONFIGS["PKU-Alignment/PKU-SafeRLHF-better"](sanity_check=True).get_preference_dataset(split="test")
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    dataset   = dataset.map(RewardDataMapFunc(tokenizer=tokenizer), batched=True)
    batch     = RewardDataCollatorWithPadding(tokenizer=tokenizer)([dataset[0], dataset[1]])
