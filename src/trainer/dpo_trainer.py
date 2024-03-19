import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from accelerate.utils import is_deepspeed_available
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments, AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from trl.import_utils import is_peft_available, is_wandb_available
from trl.models import PreTrainedModelWrapper, create_reference_model
from trl.trainer.utils import PeftSavingCallback, compute_accuracy, disable_dropout_in_model, pad_to_length

from src.utils import print_local_main, prepare_model_for_peft, get_batch_logps, pad_labels, common_prefix_length, PeftAsPreTrained


if is_peft_available():
    from peft import PeftModel

if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed


@dataclass
class DPODataMapFunc:
    """Map raw texts to tokens, attention masks, and labels."""
    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: Optional[int] = -100
    completion_only: Optional[bool] = True

    def __call__(self, examples):
        new_examples = {
            "prompt_chosen_input_ids": [],
            "prompt_chosen_attention_mask": [],
            "prompt_chosen_labels": [],

            "prompt_rejected_input_ids": [],
            "prompt_rejected_attention_mask": [],
            "prompt_rejected_labels": [],

            "prompt_input_ids": [],
            "prompt_attention_mask": [],

            "prompt": [],
        }

        for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            prompt_tokens = self.tokenizer(prompt)
            prompt_chosen_tokens = self.tokenizer(prompt + chosen)
            prompt_rejected_tokens = self.tokenizer(prompt + rejected)
            # add EOS to response
            prompt_chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            prompt_chosen_tokens["attention_mask"].append(1)
            prompt_rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            prompt_rejected_tokens["attention_mask"].append(1)

            prompt_len = common_prefix_length(prompt_chosen_tokens["input_ids"], prompt_rejected_tokens["input_ids"])

            for k, toks in {
                "prompt": prompt_tokens,
                "prompt_chosen": prompt_chosen_tokens,
                "prompt_rejected": prompt_rejected_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    new_examples[f"{k}_{type_key}"].append(tokens)
            
            for k, toks in {
                "prompt_chosen": prompt_chosen_tokens,
                "prompt_rejected": prompt_rejected_tokens,
            }.items():
                labels = toks["input_ids"].copy()
                if self.completion_only:
                    labels[:prompt_len] = [self.label_pad_token_id] * prompt_len
                new_examples[f"{k}_labels"].append(labels) 

        new_examples["prompt"] = examples["prompt"]

        return new_examples


@dataclass
class DPODataCollatorWithPadding:
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

            # `chosen` and `rejected` merged into a single batch for more efficient batched forward pass;
            right_padding_features = []
            for feature in features:
                right_padding_features.append(
                    {
                        "input_ids": feature["prompt_chosen_input_ids"],
                        "attention_mask": feature["prompt_chosen_attention_mask"],
                        "labels": feature["prompt_chosen_labels"],
                    }
                )
            for feature in features:
                right_padding_features.append(
                    {
                        "input_ids": feature["prompt_rejected_input_ids"],
                        "attention_mask": feature["prompt_rejected_attention_mask"],
                        "labels": feature["prompt_rejected_labels"],
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


class DPOTrainer(Trainer):
    r"""
    Initialize DPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        beta (`float`, defaults to 0.1):
            The beta factor in DPO loss. Higher beta means less divergence from the initial policy.
        loss_type (`str`, defaults to `"sigmoid"`):
            The type of DPO loss to use. Either `"sigmoid"` the default DPO loss or `"hinge"` loss from SLiC paper.
        args (`transformers.TrainingArguments`):
            The arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in `model` and `ref_model`.
        generate_during_eval (`bool`, defaults to `True`):
            Whether to sample and log generations during evaluation step.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
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
        filter_too_long: Optional[bool] = True,
        num_proc: Optional[int] = 4,
        generate_during_eval: bool = True,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
    ):
        if not isinstance(model, PeftModel) and is_peft_available() and peft_config:
            model = prepare_model_for_peft(model, peft_config, args)

        if ref_model:
            self.ref_model = ref_model
        elif isinstance(model, PeftModel):
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        if tokenize_map_func is None:
            tokenize_map_func = DPODataMapFunc(tokenizer, label_pad_token_id=label_pad_token_id)

        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(tokenizer, label_pad_token_id=label_pad_token_id)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
            if getattr(tokenizer, "pad_token", None) is None:
                tokenizer.pad_token = tokenizer.eos_token

        def preprocess_dataset(dataset):
            # tokenize samples
            dataset = dataset.map(
                tokenize_map_func, 
                batched=True, 
                num_proc=num_proc, 
                remove_columns=dataset.column_names
            )
            original_length = len(dataset)
            # filter samples that are too long
            if filter_too_long:
                dataset = dataset.filter(
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

        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        if generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        self.max_length = max_length
        self.generate_during_eval = generate_during_eval
        self.label_pad_token_id = label_pad_token_id

        self.beta = beta
        self.loss_type = loss_type

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

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

        if preprocess_here:
            self.log({"eval_dataset_retain": train_dataset_retain, "dataset_retain": eval_dataset_retain})

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        if self.ref_model is None:
            if not hasattr(self.model, "disable_adapter"):
                raise ValueError(
                    "You are using a `peft` version that does not support `disable_adapter`. Please update your `peft` version to the latest version."
                )
            self.ref_model = PeftAsPreTrained(self.model)
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepspeed_plugin.deepspeed_config
        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

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

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        chosen_rewards   = self.beta * (policy_chosen_logps   - reference_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)

        logits = chosen_rewards - rejected_rewards
        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(logits)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - logits)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']")

        return losses, chosen_rewards.detach(), rejected_rewards.detach()

    def forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        # batch is already concatenated
        all_logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.to(torch.float32)

        all_logps = get_batch_logps(
            all_logits,
            batch["labels"],
            average_log_prob=False,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps,  rejected_logps  = all_logps.chunk(2)
        chosen_logits, rejected_logits = all_logits.chunk(2)

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
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

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        accuracies = (chosen_rewards > rejected_rewards).float()

        # change
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

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        loss, metrics = self.get_batch_metrics(model, inputs, train_eval="train")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        policy_output = model.generate(
            input_ids=batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        reference_output = self.ref_model.generate(
            input_ids=batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, metrics = self.get_batch_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_rewards/chosen":   metrics["eval_rewards/chosen"],
            "eval_rewards/rejected": metrics["eval_rewards/rejected"],
        }
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).T.to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, np.ndarray], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value.mean())

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval and self.state.is_world_process_zero:
            # late init
            self.table = wandb.Table(columns=["Prompt", "Policy", "Ref Policy"]) if self.table == None else self.table

            print("generating response...")
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset, generate=True)
            random_batch = self._prepare_inputs(random_batch)

            policy_output_decoded, ref_output_decoded = self.get_batch_samples(self.model, random_batch)

            for prompt, policy_output, ref_policy_output in zip(random_batch["prompt"], policy_output_decoded, ref_output_decoded):
                self.table.add_data(f"(epoch{self.state.epoch}) {prompt}", policy_output[len(prompt):], ref_policy_output[len(prompt):])

        # barrier
        self.accelerator.wait_for_everyone()

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

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
    dataset   = DATASET_CONFIGS["PKU-Alignment/PKU-SafeRLHF-10K-better"](sanity_check=True).get_preference_dataset(split="train")
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    dataset   = dataset.map(DPODataMapFunc(tokenizer=tokenizer), batched=True)
    batch     = DPODataCollatorWithPadding(tokenizer=tokenizer)([dataset[0], dataset[1]])
    breakpoint()
