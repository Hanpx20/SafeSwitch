export CUDA_VISIBLE_DEVICES=5
export LLM_DIR=""
export OUTPUT_DIR=""
export TRANSFORMERS_VERBOSITY="error"
gpu_count=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

set -e


model_list=(Llama-3.1-8B-Instruct)
data_file=datasets/refusal_head_data.jsonl

for model in "${model_list[@]}"; do
    python -m src.inference.head_train \
        --model_id $LLM_DIR/$model \
        --output_dir $OUTPUT_DIR/$model \
        --train_file $data_file \
        --learning_rate 1e-5 --num_train_epochs 5
done