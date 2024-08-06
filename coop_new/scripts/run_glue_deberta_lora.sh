task=rte
log_var_init=-120
num_experts=8
vls=1e-8
normalization=Sigmoid
lr=2e-4
bs=32
sample_period=1
max_seq_length=128
for task in 'mrpc' 'rte' 'cola'
do
for num_experts in 4 8 16
do
TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0 WANDB_PROJECT='Deberta-Analysis' CUDA_VISIBLE_DEVICES=2 python examples/text-classification/run_glue.py \
--model_name_or_path microsoft/deberta-v3-base \
--expert_locations 'lora_A,lora_B' \
--num_experts ${num_experts} \
--log_variance_init ${log_var_init} \
--var_loss_scale ${vls} \
--use_entropy \
--single_variance \
--weight_normalization ${normalization} \
--expert_weight_init -1 \
--task_name ${task} \
--do_train --do_eval --max_seq_length ${max_seq_length} \
--per_device_train_batch_size ${bs} --learning_rate ${lr} \
--num_train_epochs 10 \
--evaluation_strategy steps --eval_steps 50  \
--save_strategy steps --save_steps 5000 \
--logging_steps 10 \
--report_to tensorboard \
--seed 6  \
--output_dir ../outputs/deberta/${task}/i_ex${num_experts}_lvi${log_var_init}_vls${vls}_entTrue_${normalization}_lr${lr}_bs_${bs}_loss2 \
--overwrite_output_dir \
--apply_lora \
--lora_r 8 \
--lora_alpha 16
done
done