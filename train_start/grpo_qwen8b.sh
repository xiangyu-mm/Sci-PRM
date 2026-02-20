# 设置只使用第0号卡
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
swift rlhf \
    --rlhf_type grpo \
    --model /mnt/shared-storage-user/sciprismax/public_models/Qwen3-VL-8B-Instruct \
    --external_plugins ./ms-swift/examples/train/grpo/plugin/plugin.py \
    --reward_funcs verdicts_acc \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --torch_dtype bfloat16 \
    --dataset ./toolsciverifier/sciprm-bench/train_conv_grpo_sampled.jsonl \
    --load_from_cache_file true \
    --max_length 4096 \
    --max_completion_length 4096 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 16 \
    --eval_steps 500 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --output_dir ./toolsciverifier/sciprm-bench/save_models \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --system 'You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.' \
    --log_completions true \
    --deepspeed zero2 \
    --beta 0.001 \
    --num_iterations 1
