for task in 'AddSub' 'MultiArith' 'SingleEq' 'gsm8k' 'AQuA' 'SVAMP'
do  
CUDA_VISIBLE_DEVICES=2 python evaluate.py \
 --model LLaMA-7B \
 --adapter LoRA \
 --dataset ${task} \
 --base_model 'yahma/llama-7b-hf' \
 --lora_weights 'final_trained_models/cooperative_lora/checkpoint-1800/'

CUDA_VISIBLE_DEVICES=2 python evaluate.py \
 --model LLaMA-7B \
 --adapter LoRA \
 --dataset ${task} \
 --base_model 'yahma/llama-7b-hf' \
 --lora_weights 'final_trained_models/cooperative_lora_with_loss/checkpoint-1800/'

CUDA_VISIBLE_DEVICES=2 python evaluate.py \
 --model LLaMA-7B \
 --adapter LoRA \
 --dataset ${task} \
 --base_model 'yahma/llama-7b-hf' \
 --lora_weights 'final_trained_models/cooperative_lora_without_loss/checkpoint-1800/'

done