export CUDA_VISIBLE_DEVICES=2
export LLM_DIR=/shared/nas2/shared/llms
export DATA_DIR=/shared/nas2/ph16/toxic/outputs
export TRANSFORMERS_VERBOSITY="error"


gpu_count=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
set -e

declare -A model_to_layers
model_to_layers=( ["Qwen2.5-7B-Instruct"]=28 ["Yi-1.5-9B-Chat"]=48 ["Yi-1.5-6B-Chat"]=32 ["Yi-1.5-34B-Chat"]=60 ["Phi-3-small-128k-instruct"]=32 ["Llama-3.1-8B-Instruct"]=32 ["Ministral-8B-Instruct-2410"]=36)



model_list=(Qwen2.5-7B-Instruct)


for model in "${model_list[@]}"; do
    # You can change the structure of MLP network here
    n_layers=${model_to_layers[$model]}
    if [ $model == "Yi-1.5-34B-Chat" ]; then
        hidden_size=128
    else
        hidden_size=64
    fi


    # generate model answers
    python -m src.inference.gen_answer_vllm --bench-name sorry-bench-plus \
        --model-path $LLM_DIR/$model --model-id $model  \
        --num-gpus-per-model $gpu_count
    python -m src.inference.judge_compliance_vllm --bench-name sorry-bench-plus --model-id $model


    # generate hidden states and judgements
    python -m src.prober.generate_hidden_states \
        --job_name "$model" \
        --question_file datasets/sorry-bench-plus.jsonl \
        --model_name_or_path "$LLM_DIR/$model" --batch_size 8

    python -m src.prober.generate_hidden_states \
        --job_name "$model" \
        --question_file datasets/sorry-bench-plus.jsonl \
        --model_name_or_path "$LLM_DIR/$model" --batch_size 8 \
        --do_decoding --answer_file model_answers/"$model"/sorry-bench-plus/base.jsonl

    python -m src.prober.generate_judgements \
        --job_name "$model" \
        --judgement_file model_answers/"$model"/sorry-bench-plus/judge/base/judgements.jsonl \
        --question_file datasets/sorry-bench-plus.jsonl

    python -m src.data.split_train_eval $DATA_DIR/states/"$model"


    # Direct prober with different layers
    for param in $(seq 2 2 $n_layers); do
        python -m src.prober.train_classifier \
            --data_dir $DATA_DIR/states/"$model" \
            --hidden_sizes $hidden_size --layer_id $param \
            --learning_rate 1e-5 --epochs 20 --llm $model --overwrite
    done


    # Two-stage prober, train separately with different tokens
    for param in $(seq 0 1 10); do
        python -m src.prober.train_classifier \
            --data_dir $DATA_DIR/states/"$model" \
            --hidden_sizes $hidden_size --layer_id -1 \
            --learning_rate 1e-5 --epochs 20 --llm $model --label safety \
            --token_rule multi --n_decode $param --overwrite

        python -m src.prober.train_classifier \
            --data_dir $DATA_DIR/states/"$model" \
            --hidden_sizes $hidden_size --layer_id -1 \
            --learning_rate 1e-5 --epochs 20 --llm $model --label response \
            --token_rule multi --n_decode $param --overwrite
    done


    # Evaluate two-stage probers
    for param in $(seq 0 1 10); do
        python -m src.prober.evaluate_classifier \
            --data_dir $DATA_DIR/states/"$model" --two_stage \
            --judges $DATA_DIR/classifier/"$model"_multi_safety/token0 $DATA_DIR/classifier/"$model"_multi_response/token"$param" \
            --hidden_sizes $hidden_size --overwrite --layer_id -1 --token_rule multi --n_decode $param
    done
done