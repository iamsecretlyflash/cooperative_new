tasks=('mrpc' 'cola' 'rte' 'stsb' 'sst2' 'qnli' 'mnli' 'qqp')
eval_st=(115 268 156 180 2105 3274 12272 11371)
num_experts=4
lr=6e-4
batch=(32 32 16 32 32 32 32 32)
sample_period=1
max_seq_length=128
vls=5e-3
kl_loss_weight=1e-5
for i in "${!tasks[@]}";
do
task="${tasks[$i]}"
eval_steps="${eval_st[$i]}"
bs="${batch[$i]}"
# TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0 WANDB_PROJECT='Deberta-Cooperative-Analysis' CUDA_VISIBLE_DEVICES=2 python /home/arinjay/Cooperative3/Cooperative_LLM/examples/text-classification/run_glue.py \
# --model_name_or_path microsoft/deberta-v3-base \
# --expert_locations 'query,key,value,intermediate,attention.output,layer.output' \
# --num_experts ${num_experts} \
# --var_loss_scale ${vls} \
# --use_entropy \
# --sample_period ${sample_period} \
# --kl_loss_weight ${kl_loss_weight} \
# --task_name ${task} \
# --do_train --do_eval --max_seq_length ${max_seq_length} \
# --per_device_train_batch_size ${bs} --learning_rate ${lr} \
# --num_train_epochs 5 \
# --evaluation_strategy steps --eval_steps ${eval_steps} \
# --save_strategy steps --save_steps 5000 \
# --logging_steps 10 \
# --report_to wandb \
# --seed 6  \
# --output_dir ../outputs/deberta/${task}/cooperative/ \
# --overwrite_output_dir

# TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0 WANDB_PROJECT='Deberta-Cooperative-Analysis' CUDA_VISIBLE_DEVICES=0 python /home/arinjay/Cooperative3/Cooperative_LLM/examples/text-classification/run_glue.py  \
# --model_name_or_path microsoft/deberta-v3-base \
# --expert_locations '' \
# --num_experts ${num_experts} \
# --var_loss_scale ${vls} \
# --use_entropy \
# --sample_period ${sample_period} \
# --kl_loss_weight ${kl_loss_weight} \
# --task_name ${task} \
# --do_train --do_eval --max_seq_length ${max_seq_length} \
# --per_device_train_batch_size ${bs} --learning_rate ${lr} \
# --num_train_epochs 5 \
# --evaluation_strategy steps --eval_steps ${eval_steps}  \
# --save_strategy steps --save_steps 5000 \
# --logging_steps 10 \
# --report_to wandb \
# --seed 6  \
# --output_dir ../outputs/deberta/${task}/ \
# --overwrite_output_dir

TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 WANDB_PROJECT='Deberta-Cooperative-Analysis' CUDA_VISIBLE_DEVICES=2 python /home/arinjay/Cooperative3/Cooperative_LLM/examples/text-classification/run_glue.py  \
--model_name_or_path microsoft/deberta-v3-base \
--expert_locations 'lora_A' \
--num_experts ${num_experts} \
--var_loss_scale ${vls} \
--use_entropy \
--sample_period ${sample_period} \
--kl_loss_weight ${kl_loss_weight} \
--task_name ${task} \
--do_train --do_eval --max_seq_length ${max_seq_length} \
--per_device_train_batch_size ${bs} --learning_rate ${lr} \
--num_train_epochs 10 \
--evaluation_strategy steps --eval_steps ${eval_steps}  \
--save_strategy steps --save_steps 5000 \
--logging_steps 10 \
--report_to wandb \
--seed 6  \
--apply_lora \
--lora_r 8 \
--lora_alpha 16 \
--output_dir ../outputs/deberta/${task}/cooperative_lora/10_epochs/8/16/${lr}/new/inverse \
--overwrite_output_dir

# TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0 WANDB_PROJECT='Deberta-Cooperative-Analysis' CUDA_VISIBLE_DEVICES=0 python /home/arinjay/Cooperative3/Cooperative_LLM/examples/text-classification/run_glue.py  \
# --model_name_or_path microsoft/deberta-v3-base \
# --expert_locations '' \
# --num_experts ${num_experts} \
# --var_loss_scale ${vls} \
# --use_entropy \
# --sample_period ${sample_period} \
# --kl_loss_weight ${kl_loss_weight} \
# --task_name ${task} \
# --do_train --do_eval --max_seq_length ${max_seq_length} \
# --per_device_train_batch_size ${bs} --learning_rate ${lr} \
# --num_train_epochs 5 \
# --evaluation_strategy steps --eval_steps ${eval_steps}  \
# --save_strategy steps --save_steps 5000 \
# --logging_steps 10 \
# --report_to wandb \
# --seed 6  \
# --output_dir ../outputs/deberta/${task}/lora/ \
# --overwrite_output_dir \
# --apply_lora \
# --lora_r 8 \
# --lora_alpha 16

done