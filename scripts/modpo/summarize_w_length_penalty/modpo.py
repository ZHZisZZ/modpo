import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import tyro
from accelerate import Accelerator
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, PreTrainedTokenizerBase

from src.trainer.modpo_trainer import MODPOTrainer
from src.data.configs import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE
from src.utils import print_local_main, disable_progress_bar_non_local_main, param_sharding_enabled
from src.utils.reward import RewardWrapperList, RewardWrapperBase, RewardWrapperInput

disable_progress_bar_non_local_main()


@dataclass
class ScriptArguments:

    sft_model_name: str = field(metadata={"help": "the sft model name"})
    use_flash_attention_2: Optional[bool] = field(default=False, metadata={"help": "whether to use flash attention 2"})
    prompt_template: Optional[str] = field(default=DEFAULT_PROMPT_TEMPLATE, metadata={"help": "the prompt template"})
    dataset_name: Optional[str] = field(default="Anthropic/hh-rlhf", metadata={"help": "the dataset name"})
    dataset_caching: Optional[bool] = field(default=False, metadata={"help": "used cached dataset"})
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "whether to conduct sanity check"})

    w: Optional[float] = field(default=0.5, metadata={"help": "weight"})
    beta: Optional[float] = field(default=0.1, metadata={"help": "beta for kl control"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    num_proc: Optional[int] = field(default=4, metadata={"help": "num_proc for dataset.map"})
    generate_during_eval: Optional[bool] = field(default=True, metadata={"help": "whether to generate during evaluation"})

    training_args: TrainingArguments = field(
        default_factory=lambda: TrainingArguments(
            output_dir="./output/dev/modpo",
            overwrite_output_dir=True,

            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            warmup_steps=0.1,
            weight_decay=0.05,
            fp16=True,
            remove_unused_columns=False,
            run_name="dev_modpo",
            report_to="wandb",

            num_train_epochs=3,
            logging_steps=10,
            save_steps=0.25,
            eval_steps=0.25,
            eval_delay=0.25,
            evaluation_strategy="steps",
            save_total_limit=3,
            load_best_model_at_end=True,
        )
    )

    peft: Optional[bool] = field(default=True, metadata={"help": "whether to use peft for training"})
    peft_config: LoraConfig = field(
        default_factory=lambda: LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    )

script_args = tyro.cli(ScriptArguments)
if not script_args.peft:
    script_args.peft_config = None

# base model
print_local_main("loading model...")
sft_model = AutoModelForCausalLM.from_pretrained(
    script_args.sft_model_name,
    use_flash_attention_2=script_args.use_flash_attention_2, # flash attn
    torch_dtype=torch.bfloat16, # necessary for llama2, otherwise will be cast to float32
    **({"device_map": {"": Accelerator().local_process_index}} if not param_sharding_enabled() else {}),
)
sft_model.config.update({
    "use_cache": False,
    "pad_token_id": sft_model.config.eos_token_id 
})
print_local_main(sft_model)
print_local_main(script_args.peft_config)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(script_args.sft_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# dataset
if not script_args.dataset_caching:
    from datasets import disable_caching
    disable_caching()
rdp = DATASET_CONFIGS[script_args.dataset_name](
    prompt_template=script_args.prompt_template,
    sanity_check=script_args.sanity_check,
)
train_dataset = rdp.get_preference_dataset(split="train")
eval_dataset  = rdp.get_preference_dataset(split="validation")

# get ready for training
print_local_main("start training...")
trainer = MODPOTrainer(
    model=sft_model,
    beta=script_args.beta,
    args=script_args.training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    peft_config=script_args.peft_config,
    max_length=script_args.max_length,
    num_proc=script_args.num_proc,
    generate_during_eval=script_args.generate_during_eval,
)
if Accelerator().is_local_main_process and script_args.peft_config:
    trainer.model.print_trainable_parameters()

@dataclass
class LengthPenaltyWrapper(RewardWrapperBase):
    """Penalizes longer responses."""
    tokenizer: PreTrainedTokenizerBase
    def __call__(self, inputs: RewardWrapperInput) -> torch.Tensor:
        from src.utils import prepare_input
        tokenized_responses = self.tokenizer(inputs.response)
        rewards = [-len(tokenized_response_id) for tokenized_response_id in tokenized_responses["input_ids"]]
        return prepare_input(torch.Tensor(rewards).to(torch.bfloat16))

trainer.set_wrapped_margin_reward_model_list(
    RewardWrapperList([LengthPenaltyWrapper(tokenizer=tokenizer)]),
    w=(1, script_args.w),
    prepare=False,
)
trainer.train()

save_name = "best_checkpoint" if script_args.training_args.load_best_model_at_end else "final_checkpoint"
trainer.model.save_pretrained(os.path.join(script_args.training_args.output_dir, save_name))
trainer.tokenizer.save_pretrained(os.path.join(script_args.training_args.output_dir, save_name))
