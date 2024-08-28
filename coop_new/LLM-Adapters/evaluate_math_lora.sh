export CUDA_VISIBLE_DEVICES=2
python /home/arinjay/fix/cooperative_new/coop_new/LLM-Adapters/evaluate.py \
--model LLaMA-7B \
--adapter LoRA \
--dataset AQuA \
--base_model '/home/models/llama-7b-hf' \
--lora_weights '/home/arinjay/fix/cooperative_new/coop_new/LLM-Adapters/final_trained_models/math/llama/base/3e-4/checkpoint-1839'
