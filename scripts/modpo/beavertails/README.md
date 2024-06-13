# Safety Alignment

This directory demonstrates the use of MODPO to train language models that balance helpfulness and safety. For a comprehensive experimental setup, refer to the [MODPO paper](https://arxiv.org/pdf/2310.03708.pdf).

## Automatic Pipeline

To train and evaluate language models with varying balances of helpfulness and safety:

1. Training:
```
sh scripts/modpo/beavertails/run.sh
```

2. Evaluation:
```
sh scripts/modpo/beavertails/utils/gen.py
sh scripts/modpo/beavertails/utils/score.py
```

View the results in ``output/PKU-Alignment/PKU-SafeRLHF-10K/modpo/score.csv``. Note that higher mean reward indicate better helpfulness, whereas higher mean cost suggest increased harmfulness.


## Annotated Pipeline

### Training

Two steps to train a language model that is both helpful and safe:

1. Reward Modeling: Run DPO on safe preferences to train a safe reward model that encourages safe response:
```sh
PYTHONPATH=. accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8 \
    scripts/examples/dpo/dpo.py \
    --sft_model_name "PKU-Alignment/alpaca-7b-reproduced" \
    --prompt_template "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:" \
    --dataset_name "PKU-Alignment/PKU-SafeRLHF-10K-safer" \
    --max_length 512 \
    --training_args.output_dir "./output/PKU-Alignment/PKU-SafeRLHF-10K/modpo/rm/safer" \
    --training_args.run_name "PKU-Alignment/PKU-SafeRLHF-10K/modpo/rm/safer" \
    --training_args.per_device_train_batch_size 6 \
    --training_args.per_device_eval_batch_size 6 \
    --training_args.gradient_accumulation_steps 2 \
    --training_args.learning_rate 5e-4 \
    --peft_config.r 64 \
    --peft_config.target_modules q_proj k_proj v_proj o_proj \
    --peft_config.lora_alpha 1 \
    --peft_config.lora_dropout 0
```

2. Language Modeling: Run MODPO on helpful preferences, with the safe reward as margin, to train a language model that is both helpful and safe:
```sh
PYTHONPATH=. accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8 \
    scripts/modpo/beavertails/modpo.py \
    --sft_model_name "PKU-Alignment/alpaca-7b-reproduced" \
    --margin_reward_model_name "./output/PKU-Alignment/PKU-SafeRLHF-10K/modpo/rm/safer/best_checkpoint" \
    --prompt_template "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:" \
    --dataset_name "PKU-Alignment/PKU-SafeRLHF-10K-better" \
    --max_length 512 \
    --w 0.5 \
    --training_args.output_dir "./output/PKU-Alignment/PKU-SafeRLHF-10K/modpo/lm/(0.5)better+(1-0.5)safer" \
    --training_args.run_name "PKU-SafeRLHF-10K/modpo/lm/(0.5)better+(1-0.5)safer" \
    --training_args.per_device_train_batch_size 6 \
    --training_args.per_device_eval_batch_size 6 \
    --training_args.gradient_accumulation_steps 2 \
    --training_args.learning_rate 5e-4 \
    --peft_config.r 64 \
    --peft_config.target_modules q_proj k_proj v_proj o_proj \
    --peft_config.lora_alpha 1 \
    --peft_config.lora_dropout 0
```


### Evaluation

Two steps to evaluate the trained language model:

1. Generation:
    - Generate with one GPU:
    ```sh
    # dataset can be either "PKU-Alignment/PKU-SafeRLHF-10K-safer" or "PKU-Alignment/PKU-SafeRLHF-10K-better"; only prompts are used here for generation.
    PYTHONPATH=. python3 scripts/modpo/beavertails/utils/gen.py \
        --sft_model_name "PKU-Alignment/alpaca-7b-reproduced" \
        --adapter_model_name "./output/PKU-Alignment/PKU-SafeRLHF-10K/modpo/lm/(0.5)better+(1-0.5)safer/best_checkpoint" \
        --prompt_template "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:" \
        --dataset_name "PKU-Alignment/PKU-SafeRLHF-10K-safer" \
        --output_dir "./output/PKU-Alignment/PKU-SafeRLHF-10K/modpo/lm/(0.5)better+(1-0.5)safer/gen" \
        --max_length 512
    ```
    - Generate with multiple GPUs in parallel (2 GPUs for example):
    ```sh
    for rank in 0 1; do
        CUDA_VISIBLE_DEVICES=${rank}  PYTHONPATH=. python3 scripts/modpo/beavertails/utils/gen.py \
            --sft_model_name "PKU-Alignment/alpaca-7b-reproduced" \
            --adapter_model_name "./output/PKU-Alignment/PKU-SafeRLHF-10K/modpo/lm/(0.5)better+(1-0.5)safer/best_checkpoint" \
            --prompt_template "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:" \
            --dataset_name "PKU-Alignment/PKU-SafeRLHF-10K-safer" \
            --output_dir "./output/PKU-Alignment/PKU-SafeRLHF-10K/modpo/lm/(0.5)better+(1-0.5)safer/gen" \
            --max_length 512 \
            --rank ${rank} \
            --world_size 2 &
    done

    ```

2. Scoring:
```sh
PYTHONPATH=. python3 scripts/modpo/beavertails/utils/score.py \
    --input_dir "./output/PKU-Alignment/PKU-SafeRLHF-10K/modpo/lm/(0.5)better+(1-0.5)safer/gen" \
    --output_dir "./output/PKU-Alignment/PKU-SafeRLHF-10K/modpo/lm/(0.5)better+(1-0.5)safer/score"
```
