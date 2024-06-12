import os
import inspect
from dataclasses import dataclass
from contextlib import contextmanager
from collections.abc import Mapping
from typing import Optional, Text, Any

import torch
import numpy as np
from peft import PeftModel
from accelerate import Accelerator

from trl.import_utils import is_peft_available


if is_peft_available():
    from peft import get_peft_model, prepare_model_for_kbit_training


def prepare_model_for_peft(model, peft_config, args):
    if not is_peft_available() and peft_config is not None:
        raise ValueError(
            "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
        )
    elif is_peft_available() and peft_config is not None:
        if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
            _support_gc_kwargs = hasattr(
                args, "gradient_checkpointing_kwargs"
            ) and "gradient_checkpointing_kwargs" in list(
                inspect.signature(prepare_model_for_kbit_training).parameters
            )

            preprare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

            if _support_gc_kwargs:
                preprare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

            model = prepare_model_for_kbit_training(model, **preprare_model_kwargs)
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model = get_peft_model(model, peft_config)
    # For models that use gradient_checkpoiting, we need to attach a hook that enables input
    # to explicitly have `requires_grad=True`, otherwise training will either silently
    # fail or completely fail.
    elif getattr(args, "gradient_checkpointing", False):
        # For backward compatibility with older versions of transformers
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    return model


def common_prefix_length(list_a, list_b):
    length = 0
    for i in range(min(len(list_a), len(list_b))):
        if list_a[i] == list_b[i]:
            length += 1
        else:
            break
    return length


def pad_labels(features, tokenizer, pad_to_multiple_of=None, label_pad_token_id=-100):
    # copied from https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/data/data_collator.py#L562-L584
    labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
    # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
    # same length to return tensors.
    if labels is not None:
        max_label_length = max(len(l) for l in labels)
        if pad_to_multiple_of is not None:
            max_label_length = (
                (max_label_length + pad_to_multiple_of - 1)
                // pad_to_multiple_of
                * pad_to_multiple_of
            )

        padding_side = tokenizer.padding_side
        for feature in features:
            remainder = [label_pad_token_id] * (max_label_length - len(feature["labels"]))
            if isinstance(feature["labels"], list):
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )
            elif padding_side == "right":
                feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
            else:
                feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)


def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
    label_pad_token_id: int = -100,
) -> torch.FloatTensor:
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == label_pad_token_id] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def set_adapter_ctx(model, adapter_name):
    @contextmanager
    def _set_adapter_ctx():
        old_adapter_name = model.active_adapter
        try:
            if adapter_name is not None:
                model.set_adapter(adapter_name)
                yield model
            else:
                with model.disable_adapter():
                    yield model
        finally:
            model.set_adapter(old_adapter_name)
    return _set_adapter_ctx


@dataclass
class PeftAsPreTrained:
    model: PeftModel
    adapter_name: Optional[Text] = None

    def __post_init__(self):
        assert isinstance(self.model, PeftModel)
        if self.adapter_name:
            self.ctx = set_adapter_ctx(self.model, self.adapter_name)
        else:
            self.ctx = self.model.disable_adapter

    def __call__(self, *args, **kwargs):
        with self.ctx():
            outputs = self.model(*args, **kwargs)
        return outputs

    def generate(self, *args, **kwargs):
        with self.ctx():
            outputs = self.model.generate(*args, **kwargs)
        return outputs

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.model, name)


@Accelerator().on_local_main_process
def print_local_main(text):
    print(text)


def disable_progress_bar_non_local_main():
    if not Accelerator().is_local_main_process:
        import datasets
        import transformers
        import warnings
        datasets.utils.logging.disable_progress_bar()
        transformers.utils.logging.disable_progress_bar()
        warnings.filterwarnings('ignore')


def param_sharding_enabled():
    from transformers.modeling_utils import is_deepspeed_zero3_enabled, is_fsdp_enabled
    return is_deepspeed_zero3_enabled() or is_fsdp_enabled()


def prepare_input(data):
    # adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L2626
    if isinstance(data, Mapping):
        return type(data)({k: prepare_input(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_input(v) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": Accelerator().device}
        # TODO: inference-time deepspeed?
        # if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
        #     # NLP models inputs are int/uint and those get adjusted to the right dtype of the
        #     # embedding. Other models such as wav2vec2's inputs are already float and thus
        #     # may need special handling to match the dtypes of the model
        #     kwargs.update({"dtype": Accelerator().state.deepspeed_plugin.hf_ds_config.dtype()})
        return data.to(**kwargs)
    return data


def set_seeds(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
