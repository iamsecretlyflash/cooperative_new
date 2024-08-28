tasks=('mrpc' 'cola' 'rte' 'wic' 'boolq')
eval_st=(115 268 78 170 295)
batch=(32 32 32 32 32)
num_experts=4
lr=9e-4
sample_period=1
max_seq_length=256
vls=5e-3
kl_loss_weight=1e-5
num_epochs=10
num_coop_epochs=10
num_std_epochs=0
lora_r=8
lora_alpha=16
for i in "${!tasks[@]}";
do
task="${tasks[$i]}"
eval_steps="${eval_st[$i]}"
bs="${batch[$i]}"
TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 WANDB_PROJECT='Roberta-Cooperative-PostHoc' CUDA_VISIBLE_DEVICES=2 python /home/arinjay/fix/cooperative_new/coop_new/examples/text-classification/run_glue.py \
--model_name_or_path roberta-base \
--expert_locations 'lora_A' \
--num_experts ${num_experts} \
--var_loss_scale ${vls} \
--use_entropy \
--sample_period ${sample_period} \
--kl_loss_weight ${kl_loss_weight} \
--task_name ${task} \
--do_train --do_eval --max_seq_length ${max_seq_length} \
--per_device_train_batch_size ${bs} --learning_rate ${lr} \
--num_train_epochs ${num_epochs} \
--num_coop_epochs ${num_coop_epochs} \
--num_std_epochs ${num_std_epochs} \
--evaluation_strategy steps --eval_steps ${eval_steps} \
--save_strategy steps --save_steps 5000 \
--logging_steps 10 \
--report_to wandb \
--seed 6  \
--apply_lora \
--lora_r 8 \
--lora_alpha 16 \
--posthoc_app 0 \
--output_dir ../outputs/roberta/${task}/cooperative_lora/posthoc/${num_epochs}/${num_coop_epochs}/${vls}/${lr}/${lora_r}/${lora_alpha}/${num_experts} \
--overwrite_output_dir
done
