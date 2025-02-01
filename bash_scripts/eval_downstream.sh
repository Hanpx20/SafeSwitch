#!/bin/bash
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <output_directory> <dataset_path>"
    exit 1
fi

model_id=$1
bench_name=$2
mitigate_method=$3
conditional_mitigate=$4

if [[ $bench_name == *sorry* || $bench_name == *trustllm* ]]; then
    echo "Running compliance evaluation script..."
    python -m src.inference.judge_compliance_vllm --bench-name $bench_name --model-id $model_id --mitigate_method $mitigate_method --conditional_mitigate $conditional_mitigate
elif [[ $bench_name == *alpaca* ]]; then
    # We run 3 times and get the average win rate
    echo "Running Alpaca-eval evaluation script..."

    # two "parts" are suffixs representing different runs
    if [[ "$mitigate_method" != "no" ]]; then
        mitigate_part="${mitigate_method}"
    else
        mitigate_part="base"
    fi
    if [[ "$conditional_mitigate" == "True" ]]; then
        cond_part="_cond"
    elif [[ "$conditional_mitigate" == "Prob" ]]; then
        cond_part="_prob"
    else
        cond_part=""
    fi
    

    judge_dir=model_answers/"$model_id"/$bench_name/judge/"$mitigate_part""$cond_part"
    mkdir -p $judge_dir
    final_leaderboard="$judge_dir/leaderboard.csv"
    > "$final_leaderboard"

    for ((i=1; i<=3; i++)); do
        output_dir="$judge_dir/$i"

        alpaca_eval --model_outputs "model_answers/$model_id/$bench_name/${mitigate_part}${cond_part}.json" \
                    --annotators_config "alpaca_eval_gpt4_turbo_fn" \
                    --precomputed_leaderboard None \
                    --output_path "$output_dir" \
                    --caching_path None
        
        if [[ $i -eq 1 ]]; then
            cat "$output_dir/alpaca_eval_gpt4_turbo_fn/leaderboard.csv" >> "$final_leaderboard"
        else
            tail -n +2 "$output_dir/alpaca_eval_gpt4_turbo_fn/leaderboard.csv" >> "$final_leaderboard"
        fi
        
        rm -r "$output_dir"
    done

    python -m src.inference.get_alpaca_score "$judge_dir"

elif [[ $bench_name == trivia_qa ]]; then
    echo "Running TriviaQA evaluation script..."
    python -m src.inference.eval_"$bench_name" --bench-name $bench_name --model-id $model_id --mitigate_method $mitigate_method --conditional_mitigate $conditional_mitigate
else
    echo "Unknown dataset type. Please provide a valid dataset name."
    exit 1
fi
