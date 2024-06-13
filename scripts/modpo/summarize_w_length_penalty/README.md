# Summarize with Length Penalty

This directory illustrates how we use MODPO to reduce length bias (reduce response verbosity) in summarization on the [TL;DR dataset](https://huggingface.co/datasets/openai/summarize_from_feedback).
This is a simplified version of the long-form QA experiments from the [MODPO paper](https://arxiv.org/pdf/2310.03708.pdf).

## Automatic Pipeline

To optimize a language model for human preferences with length penalty, run: 
```
sh scripts/modpo/summarize_w_length_penalty/run.sh
```


## Annotated Pipeline

1. Supervised Fine-Tuning:
    1. Run SFT with LoRA:
    ```sh
    PYTHONPATH=. accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8 \
        scripts/examples/sft/sft.py \
        --base_model_name "meta-llama/Llama-2-7b-hf" \
        --prompt_template "\n\nParagraph:\n{raw_prompt}\n\nTL;DR:\n" \
        --dataset_name "openai/summarize_from_feedback" \
        --max_length 512 \
        --chosen_only True \
        --training_args.output_dir "./output/openai/summarize_from_feedback/sft" \
        --training_args.run_name "openai/summarize_from_feedback/sft" \
        --training_args.num_train_epochs 1 \
        --training_args.per_device_train_batch_size 3 \
        --training_args.per_device_eval_batch_size 6 \
        --training_args.gradient_accumulation_steps 4 \
        --training_args.learning_rate 5e-4 \
        --peft_config.r 64 \
        --peft_config.target_modules q_proj k_proj v_proj o_proj \
        --peft_config.lora_alpha 1 \
        --peft_config.lora_dropout 0.05
    ```
    2. Merge SFT LoRA weights:
    ```sh
    PYTHONPATH=. python src/tools/merge_peft_adapter.py \
        --adapter_model_name "./output/openai/summarize_from_feedback/sft/best_checkpoint" \
        --base_model_name "meta-llama/Llama-2-7b-hf" \
        --dtype bf16 \
        --output_name "./output/openai/summarize_from_feedback/sft/merged_checkpoint"
    ```

2. Language Modeling: Run MODPO on human preferences, with length penalty as margin:
```sh
PYTHONPATH=. accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8 \
    scripts/modpo/summarize_w_length_penalty/modpo.py \
    --sft_model_name "./output/openai/summarize_from_feedback/sft/merged_checkpoint" \
    --prompt_template "\n\nParagraph:\n{raw_prompt}\n\nTL;DR:\n" \
    --dataset_name "openai/summarize_from_feedback" \
    --w 0.1 \
    --max_length 512 \
    --training_args.output_dir "./output/openai/summarize_from_feedback/modpo/lm/preference+(0.1)*length_penalty" \
    --training_args.run_name "openai/summarize_from_feedback/modpo/lm/preference+(0.1)*length_penalty" \
    --training_args.num_train_epochs 2 \
    --training_args.per_device_train_batch_size 3 \
    --training_args.per_device_eval_batch_size 6 \
    --training_args.gradient_accumulation_steps 4 \
    --training_args.learning_rate 5e-4 \
    --peft_config.r 64 \
    --peft_config.target_modules q_proj k_proj v_proj o_proj \
    --peft_config.lora_alpha 1 \
    --peft_config.lora_dropout 0
```