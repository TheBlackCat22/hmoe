model=outputs/hmoe
num_gpus=1
minibatch_size=1
grad_accum_steps=4

run_command="accelerate launch \
--mixed_precision bf16 \
--num_processes $num_gpus \
--num_machines 1 \
--gpu_ids all \
--dynamo_backend no \
--use_deepspeed \
--zero_stage 2 \
--offload_optimizer_device none \
--offload_param_device none \
--gradient_accumulation_steps $grad_accum_steps \
src/pretraining.py \
--model_name_or_path $model \
--trust_remote_code True \
--torch_dtype bfloat16 \
--max_length 512 \
--output_dir outputs/pretraining \
--overwrite_output_dir True \
--eval_strategy no \
--per_device_train_batch_size $minibatch_size \
--per_device_eval_batch_size $minibatch_size \
--gradient_accumulation_steps $grad_accum_steps \
--learning_rate 1.0e-6 \
--max_steps 10000 \
--warmup_ratio 0.1 \
--lr_scheduler_type cosine \
--logging_strategy steps \
--logging_first_step True \
--logging_steps 1 \
--save_strategy steps \
--save_steps 500 \
--save_total_limit 1 \
--save_only_model True \
--seed 42 \
--bf16 True \
--report_to tensorboard \
--ddp_find_unused_parameters False \
--gradient_checkpointing False
"
$run_command