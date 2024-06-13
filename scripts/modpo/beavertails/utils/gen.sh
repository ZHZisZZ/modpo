# sh scripts/modpo/beavertails/utils/gen.sh
sft_model_name="PKU-Alignment/alpaca-7b-reproduced"
prompt_template="BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"
dataset_name="PKU-Alignment/PKU-SafeRLHF-10K"
max_length=512

for dir in $(find output/PKU-Alignment/PKU-SafeRLHF-10K/modpo/lm -mindepth 1 -maxdepth 1 -type 'd'); do
    adapter_model_name="${dir}/best_checkpoint"
    output_dir="${dir}/gen"
    echo "#### generating responses from ckpt ${adapter_model_name} ####"
    PYTHONPATH=. python3 scripts/modpo/beavertails/utils/gen.py \
        --sft_model_name ${sft_model_name} \
        --adapter_model_name ${adapter_model_name} \
        --prompt_template "${prompt_template}" \
        --dataset_name "${dataset_name}-safer" \
        --output_dir ${output_dir} \
        --max_length ${max_length}
done
