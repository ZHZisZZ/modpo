# sh scripts/modpo/beavertails/utils/score.sh

# score
for input_dir in $(find output/PKU-Alignment/PKU-SafeRLHF-10K/modpo/lm -type 'd' | grep "gen"); do
    output_dir=${input_dir/"gen"/"score"}
    echo "#### scoring responses from dir ${input_dir} ####"
    PYTHONPATH=. python3 scripts/modpo/beavertails/utils/score.py \
        --input_dir ${input_dir} \
        --output_dir ${output_dir}
done

# aggregate scores
output_path="output/PKU-Alignment/PKU-SafeRLHF-10K/modpo/score.csv"
echo "dir, mean reward, mean cost" >> ${output_path}
for input_path in $(find output/PKU-Alignment/PKU-SafeRLHF-10K/modpo/lm | grep "score/mean.csv"); do
    echo -n "${input_path}, " >> ${output_path}
    cat ${input_path} | tail -1 >> ${output_path}
done
