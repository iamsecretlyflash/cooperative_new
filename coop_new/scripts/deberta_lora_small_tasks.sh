tasks=('mrpc' 'cola')
eval_st=(115 268 )
learning_rates=(9e-4 5e-4)
coop_learning_rates=(9e-4 9e-4)
num_experts=4
lr=9e-4
batch=(32 32)
# tasks=('rte' 'stsb' )
# eval_st=(78 180 )
# learning_rates=(1.2e-3 2.2e-3)
# num_experts=4
# lr=9e-4
# batch=(32 32 )
sample_period=1
max_seq_length=300
#max seq len has been set only for mrpc
vls=5e-4
kl_loss_weight=1e-5
num_epochs=14
num_coop_epochs=10
num_std_epochs=4
lora_r=8
post=0
lora_alpha=32
for i in "${!tasks[@]}";
do
task="${tasks[$i]}"
eval_steps="${eval_st[$i]}"
bs="${batch[$i]}"
lr="${learning_rates[$i]}"
TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 WANDB_PROJECT='Deberta-Cooperative-Small-Task-Final' CUDA_VISIBLE_DEVICES=1 python /home/vaibhav/coop_headache2/cooperative_new/coop_new/examples/text-classification/run_glue.py  \
--model_name_or_path microsoft/deberta-v3-base \
--expert_locations 'lora_A' \
--num_experts ${num_experts} \
--var_loss_scale ${vls} \
--use_entropy \
--posthoc_app ${post} \
--sample_period ${sample_period} \
--kl_loss_weight ${kl_loss_weight} \
--task_name ${task} \
--do_train  \
--do_eval --max_seq_length ${max_seq_length} \
--per_device_train_batch_size ${bs} --learning_rate ${lr} \
--num_train_epochs ${num_epochs} \
--num_coop_epochs ${num_coop_epochs} \
--num_std_epochs ${num_std_epochs} \
--evaluation_strategy steps --eval_steps ${eval_steps}  \
--save_strategy steps --save_steps 5000 \
--logging_steps 10 \
--report_to wandb \
--seed 6  \
--apply_lora \
--lora_r ${lora_r} \
--lora_alpha ${lora_alpha} \
--weight_decay 0.01 \
--output_dir ../outputs/deberta/${task}/cooperative_lora/posthoc_${post}/std_${num_std_epochs}/coop_${num_coop_epochs}/${kl_loss_weight}/${lr}/${lora_r}/${lora_alpha}/${num_experts} \
--overwrite_output_dir
done

#added weight decay as it was not present before
