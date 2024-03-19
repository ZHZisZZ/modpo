# sh scripts/modpo/beavertails/run.sh
LAUNCH="accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8"

sft_model_name="PKU-Alignment/alpaca-7b-reproduced"
prompt_template="BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"
dataset_name="PKU-Alignment/PKU-SafeRLHF-10K"
sanity_check=False
output_dir="./output"
max_length=512
per_device_train_batch_size=6
per_device_eval_batch_size=6
gradient_accumulation_steps=2
learning_rate=5e-4

# Note that `PKU-Alignment/alpaca-7b-reproduced` is a SFTed model in the first place; therefore, there is no need to SFT again. 

# MODPO stage 1: margin reward modeling for safety
# Note that we parametrize this reward model implicitly as a language model: r(x,y) = logp(y|x)-logp_{sft}(y|x) and train with DPO.
mrm_run_name="${dataset_name}/modpo/mrm-safer"
PYTHONPATH=. $LAUNCH scripts/examples/dpo/dpo.py \
    --sft_model_name ${sft_model_name} \
    --prompt_template "${prompt_template}" \
    --dataset_name "${dataset_name}-safer" \
    --sanity_check ${sanity_check} \
    --max_length ${max_length} \
    --training_args.output_dir "${output_dir}/${mrm_run_name}" \
    --training_args.run_name ${mrm_run_name} \
    --training_args.per_device_train_batch_size ${per_device_train_batch_size} \
    --training_args.per_device_eval_batch_size ${per_device_eval_batch_size} \
    --training_args.gradient_accumulation_steps ${gradient_accumulation_steps} \
    --training_args.learning_rate ${learning_rate} \
    --peft_config.r 64 \
    --peft_config.target_modules q_proj k_proj v_proj o_proj \
    --peft_config.lora_alpha 1 \
    --peft_config.lora_dropout 0

# MODPO stage 2: language modeling with the safety reward as margin
# r = (w)*r_better + (1-w)*r_safer
w=0.5
lm_run_name="${dataset_name}/modpo/lm/($w)*r_better+(1-$w)*r_safe"
PYTHONPATH=. $LAUNCH scripts/modpo/beavertails/modpo.py \
    --sft_model_name ${sft_model_name} \
    --margin_reward_model_name "${output_dir}/${mrm_run_name}/best_checkpoint" \
    --prompt_template "${prompt_template}" \
    --dataset_name "${dataset_name}-better" \
    --sanity_check ${sanity_check} \
    --w ${w} \
    --max_length ${max_length} \
    --training_args.output_dir "${output_dir}/${lm_run_name}" \
    --training_args.run_name ${lm_run_name} \
    --training_args.per_device_train_batch_size ${per_device_train_batch_size} \
    --training_args.per_device_eval_batch_size ${per_device_eval_batch_size} \
    --training_args.gradient_accumulation_steps ${gradient_accumulation_steps} \
    --training_args.learning_rate ${learning_rate} \
    --peft_config.r 64 \
    --peft_config.target_modules q_proj k_proj v_proj o_proj \
    --peft_config.lora_alpha 1 \
    --peft_config.lora_dropout 0
