from abc import ABC, abstractclassmethod
from dataclasses import dataclass, asdict
from typing import Any, Text, List, Dict, Optional

import torch
from accelerate import Accelerator
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from src.utils import get_batch_logps, prepare_input
from src.data.configs import DEFAULT_PROMPT_TEMPLATE


@dataclass
class RewardWrapperInput:
    raw_prompt: List[str]
    response: List[str]


@dataclass
class RewardWrapperBase(ABC):
    @abstractclassmethod
    def __call__(self, inputs: Any) -> Any:
        raise NotImplementedError


@dataclass
class RewardWrapperList(RewardWrapperBase):
    reward_wrapper_list: List[RewardWrapperBase]

    def __call__(self, inputs: Any) -> List[torch.Tensor]:
        outputs_list = []
        for reward_wrapper in self.reward_wrapper_list:
            outputs_list.append(reward_wrapper(inputs))
        return outputs_list

    def map(self, func):
        for i in range(len(self.reward_wrapper_list)):
            self.reward_wrapper_list[i] = func(self.reward_wrapper_list[i])
        return self

    def __len__(self):
        return len(self.reward_wrapper_list)


@dataclass
class ImplicitRewardWrapper(RewardWrapperBase):
    """
    An implicit reward model parameterized as r(x,y) = logp(y|x)-logp_{ref}(y|x)
    """
    model: PreTrainedModel
    ref_model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    prompt_template: Optional[str] = DEFAULT_PROMPT_TEMPLATE
    beta: Optional[bool] = 0.1
    average_log_prob: Optional[bool] = False
    label_pad_token_id: Optional[int] = -100

    @torch.no_grad()
    def __call__(self, inputs: RewardWrapperInput) -> torch.Tensor:
        from src.trainer.sft_trainer import SFTDataMapFunc, SFTDataCollatorWithPadding
        inputs = asdict(inputs)
        inputs["prompt"] = [self.prompt_template.format(
            raw_prompt=raw_prompt) for raw_prompt in inputs["raw_prompt"]]
        tokens = SFTDataMapFunc(tokenizer=self.tokenizer)(inputs)
        batch  = SFTDataCollatorWithPadding(tokenizer=self.tokenizer)(
            [{k:v[i] for k,v in tokens.items()} for i in range(len(inputs["prompt"]))]
        )
        batch = prepare_input(batch)
        policy_all_logps = self.forward(self.model, batch)
        ref_all_logps = self.forward(self.ref_model, batch)
        return self.beta * (policy_all_logps - ref_all_logps)

    @torch.no_grad()
    def forward(self, model: PreTrainedModel, batch: Dict[Text, torch.Tensor]) -> torch.Tensor:
        all_logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.to(torch.float32)
        all_logps = get_batch_logps(
            all_logits,
            batch["labels"],
            average_log_prob=self.average_log_prob,
            label_pad_token_id=self.label_pad_token_id,
        )
        return all_logps


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from accelerate import Accelerator

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        use_flash_attention_2=True, # flash attn
        torch_dtype=torch.bfloat16,
        device_map={"": Accelerator().local_process_index},
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    implicit_reward = ImplicitRewardWrapper(
        model=model,
        ref_model=model,
        tokenizer=tokenizer,
    )

    implicit_reward({"raw_prompt": ["who are you", "hi"], "response": ["i am your dad", "goodbye"]})
    breakpoint()
