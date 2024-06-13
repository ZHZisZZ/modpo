# sh scripts/modpo/summarize_w_length_penalty/run.sh
LAUNCH="accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8"

base_model_name="meta-llama/Llama-2-7b-hf"
prompt_template="\n\nParagraph:\n{raw_prompt}\n\nTL;DR:\n"
dataset_name="openai/summarize_from_feedback"
sanity_check=False
output_dir="./output"
max_length=512
per_device_train_batch_size=3
per_device_eval_batch_size=6
gradient_accumulation_steps=4
learning_rate=5e-4

# Supervised Fine-Tuning: Run SFT with LoRA
sft_run_name="${dataset_name}/sft"
chosen_only=True
PYTHONPATH=. $LAUNCH scripts/examples/sft/sft.py \
    --base_model_name ${base_model_name} \
    --prompt_template "${prompt_template}" \
    --dataset_name ${dataset_name} \
    --sanity_check ${sanity_check} \
    --max_length ${max_length} \
    --chosen_only ${chosen_only} \
    --training_args.output_dir "${output_dir}/${sft_run_name}" \
    --training_args.run_name ${sft_run_name} \
    --training_args.num_train_epochs 1 \
    --training_args.per_device_train_batch_size ${per_device_train_batch_size} \
    --training_args.per_device_eval_batch_size ${per_device_eval_batch_size} \
    --training_args.gradient_accumulation_steps ${gradient_accumulation_steps} \
    --training_args.learning_rate ${learning_rate} \
    --peft_config.r 64 \
    --peft_config.target_modules q_proj k_proj v_proj o_proj \
    --peft_config.lora_alpha 1 \
    --peft_config.lora_dropout 0.05

# Supervised Fine-Tuning: Merge SFT LoRA weights
PYTHONPATH=. python src/tools/merge_peft_adapter.py \
    --adapter_model_name "${output_dir}/${sft_run_name}/best_checkpoint" \
    --base_model_name ${base_model_name} \
    --dtype bf16 \
    --output_name "${output_dir}/${sft_run_name}/merged_checkpoint"

# Language Model: Run MODPO on human preferences, with length penalty as margin
# r = r_prefernce + w*length_penalty
w=0.1
lm_run_name="${dataset_name}/modpo/lm/preference+($w)*length_penalty"
PYTHONPATH=. $LAUNCH scripts/modpo/summarize_w_length_penalty/modpo.py \
    --sft_model_name "${output_dir}/${sft_run_name}/best_checkpoint" \
    --prompt_template "${prompt_template}" \
    --dataset_name "${dataset_name}" \
    --sanity_check ${sanity_check} \
    --w ${w} \
    --max_length ${max_length} \
    --training_args.output_dir "${output_dir}/${lm_run_name}" \
    --training_args.run_name ${lm_run_name} \
    --training_args.num_train_epochs 2 \
    --training_args.per_device_train_batch_size ${per_device_train_batch_size} \
    --training_args.per_device_eval_batch_size ${per_device_eval_batch_size} \
    --training_args.gradient_accumulation_steps ${gradient_accumulation_steps} \
    --training_args.learning_rate ${learning_rate} \
    --peft_config.r 64 \
    --peft_config.target_modules q_proj k_proj v_proj o_proj \
    --peft_config.lora_alpha 1 \
    --peft_config.lora_dropout 0
