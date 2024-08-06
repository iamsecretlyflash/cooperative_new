task=mrpc
TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0 CUDA_VISIBLE_DEVICES=2 python examples/text-classification/run_glue.py \
--model_name_or_path bert-base-cased \
--expert_locations query \
--num_experts 4 \
--log_variance_init -10 \
--var_loss_scale 1 \
--use_entropy True \
--task_name ${task} \
--do_train --do_eval --max_seq_length 128 \
--per_device_train_batch_size 32 --learning_rate 3e-5 \
--num_train_epochs 10 \
--evaluation_strategy steps --eval_steps 100 \
--save_strategy steps --save_steps 10000 \
--logging_steps 10 \
--report_to wandb \
--seed 6  \
--output_dir ./output/bert-base-none/${task} \
--overwrite_output_dir
