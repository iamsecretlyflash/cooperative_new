tasks=('mrpc' 'cola' 'rte' 'stsb' 'sst2' 'qnli' 'mnli' 'qqp')
eval_st=(115 268 156 180 2105 3274 12272 11371)
batch=(32 32 16 32 32 32 32 32)
num_experts=4
learn_rate=(3e-5 2e-5 3e-5 6e-5 3e-5 3e-5 3e-5 3e-5)
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
TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0 WANDB_PROJECT='Deberta-Cooperative-Analysis' CUDA_VISIBLE_DEVICES=0 python /home/vaibhav/Untitled/examples/text-classification/run_glue.py \
--model_name_or_path microsoft/deberta-v3-base \
--expert_locations 'query,key' \
--num_experts ${num_experts} \
--var_loss_scale ${vls} \
--use_entropy \
--sample_period ${sample_period} \
--kl_loss_weight ${kl_loss_weight} \
--task_name ${task} \
--do_train --do_eval --max_seq_length ${max_seq_length} \
--per_device_train_batch_size ${bs} --learning_rate ${lr} \
--num_train_epochs 15 \
--num_coop_epochs 10 \
--num_std_epochs 5 \
--evaluation_strategy steps --eval_steps ${eval_steps} \
--save_strategy steps --save_steps 5000 \
--logging_steps 10 \
--report_to wandb \
--seed 6  \
--output_dir ../outputs/deberta/${task}/query/inverse/${vls}/${lr}/${num_experts} \
--overwrite_output_dir
done




# tensor([[ 0.0786,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
#         [-0.0018,  0.0780,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
#         [-0.0012, -0.0011,  0.0782,  ...,  0.0000,  0.0000,  0.0000],
#         ...,
#         [-0.0014, -0.0015, -0.0017,  ...,  0.0013,  0.0000,  0.0000],
#         [ 0.0005, -0.0002, -0.0019,  ...,  0.0009,  0.0003,  0.0000],
#         [ 0.0013,  0.0015,  0.0010,  ..., -0.0018,  0.0012,  0.0000]],
