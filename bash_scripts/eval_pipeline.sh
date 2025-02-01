export CUDA_VISIBLE_DEVICES=5
export LLM_DIR=""
export CLASSIFIER_DIR=""
export REFUSAL_HEAD_DIR=""
export OPENAI_API_KEY=""
export TRANSFORMERS_VERBOSITY="error"
gpu_count=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

set -e

model_list=(Qwen2.5-7B-Instruct Llama-3.1-8B-Instruct Ministral-8B-Instruct-2410 Yi-1.5-9B-Chat)
bench_list=("sorry-bench-eval" "trustllm-misuse" "trustllm-jailbreak" "trustllm-exaggerated_safety" "alpaca-eval" "trivia_qa")
mitigate_method_list=(no prompt_strong head)
conditional_mitigate_list=("False" "True")

n_decode=3


for model in "${model_list[@]}"; do
    for bench in "${bench_list[@]}"; do
        for mitigate_method in "${mitigate_method_list[@]}"; do
            for conditional_mitigate in "${conditional_mitigate_list[@]}"; do
                if [[ $mitigate_method == no && $conditional_mitigate == "True" ]]; then
                    continue
                fi
                max_new_token=1024

                python -m src.inference.gen_answer_vllm --bench-name $bench --model-path "$LLM_DIR/$model" --model-id $model \
                    --max-new-token $max_new_token --num-gpus-per-model $gpu_count \
                    --mitigate_method $mitigate_method --conditional_mitigate $conditional_mitigate \
                    --judges $CLASSIFIER_DIR/"$model"_multi_safety/token0 $CLASSIFIER_DIR/"$model"_multi_response/token"$n_decode" \
                    --layer -1 --n_decode $n_decode \
                    --finetune-model-path $REFUSAL_HEAD_DIR/"$model"/final_model

                bash_scripts/eval_downstream.sh $model $bench $mitigate_method $conditional_mitigate

            done
        done
    done
done
