# sh scripts/examples/sft/run.sh
LAUNCH="accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8"

base_model_name="meta-llama/Llama-2-7b-hf"
dataset_name="PKU-Alignment/PKU-SafeRLHF-10K-safer"
max_length=512
sanity_check=False
chosen_only=False
output_dir="./output"

# SFT
sft_run_name="${dataset_name}/sft"
PYTHONPATH=. $LAUNCH scripts/examples/sft/sft.py \
    --base_model_name ${base_model_name} \
    --dataset_name ${dataset_name} \
    --sanity_check ${sanity_check} \
    --max_length ${max_length} \
    --chosen_only ${chosen_only} \
    --training_args.output_dir "${output_dir}/${sft_run_name}" \
    --training_args.run_name ${sft_run_name} \
    --peft_config.target_modules q_proj k_proj v_proj o_proj

sft_model_name="${output_dir}/${sft_run_name}/merged_checkpoint"

# Merge SFT LoRA weights
PYTHONPATH=. python src/tools/merge_peft_adapter.py \
    --adapter_model_name "${output_dir}/${sft_run_name}/best_checkpoint" \
    --base_model_name ${base_model_name} \
    --output_name ${sft_model_name}
