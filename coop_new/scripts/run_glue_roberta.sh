tasks=('cola' 'stsb' 'sst2' 'qnli' 'mnli' 'qqp')
eval_st=(268 180 2105 3274 12272 11371)
num_experts=4
lr=5e-4
bs=32
sample_period=1
max_seq_length=128
vls=5e-3
kl_loss_weight=1e-5
for i in "${!tasks[@]}";
do
task="${tasks[$i]}"
eval_steps="${eval_st[$i]}"
TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0 WANDB_PROJECT='Roberta-Cooperative-Analysis' CUDA_VISIBLE_DEVICES=0 python /home/arinjay/Cooperative3/Cooperative_LLM/examples/text-classification/run_glue.py \
--model_name_or_path roberta-base \
--expert_locations 'query,key,value,intermediate,attention.output,layer.output' \
--num_experts ${num_experts} \
--var_loss_scale ${vls} \
--use_entropy \
--sample_period ${sample_period} \
--kl_loss_weight ${kl_loss_weight} \
--task_name ${task} \
--do_train --do_eval --max_seq_length ${max_seq_length} \
--per_device_train_batch_size ${bs} --learning_rate ${lr} \
--num_train_epochs 5 \
--evaluation_strategy steps --eval_steps ${eval_steps} \
--save_strategy steps --save_steps 5000 \
--logging_steps 10 \
--report_to wandb \
--seed 6  \
--output_dir ../outputs/roberta/${task}/cooperative/ \
--overwrite_output_dir
TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0 WANDB_PROJECT='Roberta-Cooperative-Analysis' CUDA_VISIBLE_DEVICES=0 python /home/arinjay/Cooperative3/Cooperative_LLM/examples/text-classification/run_glue.py  \
--model_name_or_path roberta-base \
--expert_locations '' \
--num_experts ${num_experts} \
--var_loss_scale ${vls} \
--use_entropy \
--sample_period ${sample_period} \
--kl_loss_weight ${kl_loss_weight} \
--task_name ${task} \
--do_train --do_eval --max_seq_length ${max_seq_length} \
--per_device_train_batch_size ${bs} --learning_rate ${lr} \
--num_train_epochs 5 \
--evaluation_strategy steps --eval_steps ${eval_steps}  \
--save_strategy steps --save_steps 5000 \
--logging_steps 10 \
--report_to wandb \
--seed 6  \
--output_dir ../outputs/roberta/${task}/ \
--overwrite_output_dir
done