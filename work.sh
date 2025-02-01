export CUDA_VISIBLE_DEVICES=2
python src/safeswitch_pipeline.py --model Llama-3.1-8B-Instruct \
    --llm_dir /shared/nas2/shared/llms \
    --classifier_dir /shared/nas2/ph16/toxic/outputs/classifier \
    --refusal_head_dir /shared/nas2/ph16/toxic/finetuned_LM_head