hostfile=""
deepspeed --hostfile=$hostfile  --include=localhost:0,1,2,3  \
    fine-tune_copy.py \
    --report_to "tensorboard" \
    --data_folder  "/data/xgao/Baichuan2/fine-tune/data/filtered_corpus" \
    --model_name_or_path "THUDM/glm-4-9b" \
    --output_dir "output_3" \
    --model_max_length 1024 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --deepspeed ds_config.json \
    --bf16 True \
    --tf32 True \
    --use_lora True \
    --logging_dir "./logs"
    