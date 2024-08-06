num_experts=4
sample_period=1
max_seq_length=128
vls=5e-4
kl_loss_weight=1e-5
for exp in "key" "value" "intermediate" "attention.output" "layer.output";
do
task="mrpc"
eval_steps=115
bs=32
lr=3e-5
TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0 WANDB_PROJECT='Deberta-SST2' CUDA_VISIBLE_DEVICES=2 python /home/arinjay/Cooperative3/Cooperative_LLM/examples/text-classification/run_glue.py \
--model_name_or_path microsoft/deberta-v3-base \
--expert_locations ${exp} \
--num_experts ${num_experts} \
--var_loss_scale ${vls} \
--use_entropy \
--sample_period ${sample_period} \
--kl_loss_weight ${kl_loss_weight} \
--task_name ${task} \
--do_train --do_eval --max_seq_length ${max_seq_length} \
--per_device_train_batch_size ${bs} --learning_rate ${lr} \
--num_train_epochs 10 \
--evaluation_strategy steps --eval_steps ${eval_steps} \
--save_strategy steps --save_steps 5000 \
--logging_steps 10 \
--report_to wandb \
--seed 6  \
--output_dir ../outputs/deberta/${task}/${exp}/${lr}/${num_experts}/${sample_period}/check/inverse \
--overwrite_output_dir
done