#export CUDA_VISIBLE_DEVICES=0
#python evaluate.py \
#--model LLaMA-7B \
#--adapter LoRA \
#--dataset AQuA \
#--base_model '/home/models/llama-7b-hf' \
#--lora_weights '/home/vaibhav/MTD_New/CoopUpdated/Cooperative_LLM/LLM-Adapters/final_trained_models/cooperative_lora_with_entropy_loss/checkpoint-1800
export CUDA_VISIBLE_DEVICES=1 
python commonsense_evaluate.py \
--model LLaMA-7B \
--adapter LoRA \
--dataset boolq \
--base_model '/home/models/Llama-2-7b-hf' \
--lora_weights '/home/arinjay/Cooperative3/Cooperative_LLM/LLM-Adapters/final_trained_models/cs/lora_A/checkpoint-127600' \
--batch_size 4