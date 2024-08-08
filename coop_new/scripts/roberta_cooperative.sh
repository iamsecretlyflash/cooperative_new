tasks=('mrpc')
eval_st=(115)
batch=(32)
num_experts=4
learn_rate=(5e-5)
sample_period=1
max_seq_length=128
vls=5e-3
kl_loss_weight=1e-5
for i in "${!tasks[@]}";
do
task="${tasks[$i]}"
eval_steps="${eval_st[$i]}"
bs="${batch[$i]}"
lr="${learn_rate[$i]}"
TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0 WANDB_PROJECT='Roberta-Cooperative-Analysis' CUDA_VISIBLE_DEVICES=1 python /home/vaibhav/Untitled/examples/text-classification/run_glue.py \
--model_name_or_path roberta-base \
--expert_locations 'query' \
--num_experts ${num_experts} \
--var_loss_scale ${vls} \
--use_entropy \
--sample_period ${sample_period} \
--kl_loss_weight ${kl_loss_weight} \
--task_name ${task} \
--do_train --do_eval --max_seq_length ${max_seq_length} \
--per_device_train_batch_size ${bs} --learning_rate ${lr} \
--num_train_epochs 10 \
--num_coop_epochs 10 \
--num_std_epochs 0 \
--evaluation_strategy steps --eval_steps ${eval_steps} \
--save_strategy steps --save_steps 5000 \
--logging_steps 10 \
--report_to wandb \
--seed 6  \
--output_dir ../outputs/roberta/${task}/query/${lr}/check/inverse/b15 \
--overwrite_output_dir
done
