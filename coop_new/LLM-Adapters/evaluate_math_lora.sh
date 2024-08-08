export CUDA_VISIBLE_DEVICES=0
python evaluate.py \
--model LLaMA-7B \
--adapter LoRA \
--dataset AQuA \
--base_model '/home/models/llama-7b-hf' \
--lora_weights '/home/arinjay/Cooperative3/Cooperative_LLM/LLM-Adapters/final_trained_models/math/llama/lora/checkpoint-7200'
