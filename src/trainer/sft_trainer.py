import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers import PreTrainedTokenizerBase, TrainingArguments
from transformers.trainer_utils import EvalLoopOutput
from trl.import_utils import is_peft_available, is_wandb_available
from trl.trainer.utils import (
    PeftSavingCallback,
)

from src.utils import pad_labels, print_local_main, prepare_model_for_peft, common_prefix_length


if is_peft_available():
    from peft import PeftConfig, PeftModel


@dataclass
class SFTDataMapFunc:
    """Map raw texts to tokens, attention masks, and labels."""
    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: Optional[int] = -100
    completion_only: Optional[bool] = True

    def __call__(self, examples):
        new_examples = {
            "prompt_response_input_ids": [],
            "prompt_response_attention_mask": [],
            "prompt_response_labels": [],

            "prompt_input_ids": [],
            "prompt_attention_mask": [],

            "prompt": [],
        }
        for prompt, response in zip(examples["prompt"], examples["response"]):
            prompt_tokens = self.tokenizer(prompt)
            prompt_response_tokens = self.tokenizer(prompt + response)
            # add EOS to response
            prompt_response_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            prompt_response_tokens["attention_mask"].append(1)

            prompt_len = common_prefix_length(prompt_tokens["input_ids"], prompt_response_tokens["input_ids"])

            for k, toks in {
                "prompt": prompt_tokens,
                "prompt_response": prompt_response_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    new_examples[f"{k}_{type_key}"].append(tokens)
            
            for k, toks in {
                "prompt_response": prompt_response_tokens,
            }.items():
                labels = toks["input_ids"].copy()
                if self.completion_only:
                    labels[:prompt_len] = [self.label_pad_token_id] * prompt_len
                new_examples[f"{k}_labels"].append(labels) 

        new_examples["prompt"] = examples["prompt"]

        return new_examples


@dataclass
class SFTDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: Optional[int] = -100
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]], generate: Optional[bool] = False) -> Dict[str, Any]:
        """
        if not generate:
            batch = {
                "input_ids": ...,
                "attention_mask": ...,
                "labels": ...,
            }
        else:
            batch = {
                "prompt": ...,
                "prompt_input_ids": ...,
                "prompt_attention_mask": ...,
            }
        """
        if not generate:

            # right padding for training
            right_padding_features = []
            for feature in features:
                right_padding_features.append(
                    {
                        "input_ids": feature["prompt_response_input_ids"],
                        "attention_mask": feature["prompt_response_attention_mask"],
                        "labels": feature["prompt_response_labels"],
                    }
                )

            pad_labels(right_padding_features, self.tokenizer, self.pad_to_multiple_of, self.label_pad_token_id)

            right_padding_batch = self.tokenizer.pad(
                right_padding_features,
                padding=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            return right_padding_batch

        else:

            # left padding for batched generation
            left_padding_features = [] 
            padding_side_default = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"
            for feature in features:
                left_padding_features.append(
                    {
                        "input_ids": feature["prompt_input_ids"],
                        "attention_mask": feature["prompt_attention_mask"],
                    }
                )
            left_padding_batch = self.tokenizer.pad(
                left_padding_features,
                padding=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            self.tokenizer.padding_side = padding_side_default

            return {
                "prompt": [feature["prompt"] for feature in features],
                "prompt_input_ids": left_padding_batch["input_ids"],
                "prompt_attention_mask": left_padding_batch["attention_mask"],
            }


class SFTTrainer(Trainer):
    r"""
    Class definition of the Supervised Finetuning Trainer (SFT Trainer).
    This class is a wrapper around the `transformers.Trainer` class and inherits all of its attributes and methods.
    The trainer takes care of properly initializing the PeftModel in case a user passes a `PeftConfig` object.

    Args:
        model (Union[`transformers.PreTrainedModel`, `nn.Module`, `str`]):
            The model to train, can be a `PreTrainedModel`, a `torch.nn.Module` or a string with the model name to
            load from cache or download. The model can be also converted to a `PeftModel` if a `PeftConfig` object is
            passed to the `peft_config` argument.
        args (Optional[`transformers.TrainingArguments`]):
            The arguments to tweak for training. Please refer to the official documentation of `transformers.TrainingArguments`
            for more information.
        data_collator (Optional[`transformers.DataCollator`]):
            The data collator to use for training.
        train_dataset (Optional[`datasets.Dataset`]):
            The dataset to use for training. We recommend users to use `trl.trainer.ConstantLengthDataset` to create their dataset.
        eval_dataset (Optional[Union[`datasets.Dataset`, Dict[`str`, `datasets.Dataset`]]]):
            The dataset to use for evaluation. We recommend users to use `trl.trainer.ConstantLengthDataset` to create their dataset.
        tokenizer (Optional[`transformers.PreTrainedTokenizer`]):
            The tokenizer to use for training. If not specified, the tokenizer associated to the model will be used.
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
        peft_config (`Optional[PeftConfig]`):
            The PeftConfig object to use to initialize the PeftModel.
        max_length (`Optional[int]`):
            The maximum sequence length to use for the `ConstantLengthDataset` and for automaticallty creating the Dataset. Defaults to `512`.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        args: TrainingArguments = None,
        tokenize_map_func: Optional[Callable] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[PeftConfig] = None,
        max_length: Optional[int] = 1024,
        completion_only: Optional[bool] = True,
        num_proc: Optional[int] = 4,
        generate_during_eval: Optional[bool] = True,
    ):
        if not isinstance(model, PeftModel) and is_peft_available() and peft_config:
            model = prepare_model_for_peft(model, peft_config, args)

        if tokenize_map_func is None:
            tokenize_map_func = SFTDataMapFunc(tokenizer, completion_only=completion_only)

        if data_collator is None:
            data_collator = SFTDataCollatorWithPadding(tokenizer)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
            if getattr(tokenizer, "pad_token", None) is None:
                tokenizer.pad_token = tokenizer.eos_token

        # preprocess dataset
        def preprocess_dataset(dataset):
            # tokenize samples
            dataset  = dataset.map(
                tokenize_map_func, 
                batched=True, 
                num_proc=num_proc, 
                remove_columns=dataset.column_names
            )
            # truncate samples that are too long
            dataset = dataset.map(
                lambda sample: {k: v[:max_length] for k, v in sample.items()},
                num_proc=num_proc,
            )
            return dataset
        if "prompt_response_input_ids" not in train_dataset[0].keys():
            print_local_main("dataset preprocessing...")
            train_dataset = preprocess_dataset(train_dataset)
            eval_dataset  = preprocess_dataset(eval_dataset)

        if is_peft_available() and isinstance(model, PeftModel):
            if callbacks is None:
                callbacks = [PeftSavingCallback()]
            else:
                callbacks += [PeftSavingCallback()]

        if generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        self.max_length = max_length
        self.generate_during_eval = generate_during_eval

        self.table = None # for late initialization

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        initial_output = super().train(
            resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs,
        )

        # upload wandb table at the end of training if it exists
        if self.table:
            self.log({"eval_game_log": self.table})
            self.state.log_history.pop()

        return initial_output

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        # adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L600-L647
        if self.generate_during_eval and self.state.is_world_process_zero:
            # late init
            self.table = wandb.Table(columns=["Prompt", "Policy"]) if self.table == None else self.table

            print("generating response...")
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset, generate=True)
            random_batch = self._prepare_inputs(random_batch)

            # get batch samples
            policy_output = self.model.generate(
                input_ids=random_batch["prompt_input_ids"],
                attention_mask=random_batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            response_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

            for prompt, response in zip(random_batch["prompt"], response_decoded):
                self.table.add_data(f"(epoch{self.state.epoch}) {prompt}", response[len(prompt):])

        # barrier
        self.accelerator.wait_for_everyone()

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output


if __name__ == '__main__':
    from src.data.configs import DATASET_CONFIGS
    from transformers import LlamaTokenizer
    dataset   = DATASET_CONFIGS["PKU-Alignment/PKU-SafeRLHF-10K-better"](sanity_check=True).get_sft_dataset(split="train")
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token

    # SFTDataMapFunc unit test
    dataset = dataset.map(SFTDataMapFunc(tokenizer=tokenizer), batched=True, remove_columns=dataset.column_names)
    
    # SFTDataCollatorWithPadding unit test
    batch = SFTDataCollatorWithPadding(tokenizer=tokenizer)([dataset[0], dataset[1]])
    batch_prompt = SFTDataCollatorWithPadding(tokenizer=tokenizer)([dataset[0], dataset[1]], generate=True)
    breakpoint()
