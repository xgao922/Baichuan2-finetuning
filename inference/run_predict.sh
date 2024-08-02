#!/bin/bash

num_outputs_list=(5 10 20 50 100) # 
model_path="/home/xgao/Baichuan2/fine-tune/output_2/checkpoint-468"

for num_outputs in "${num_outputs_list[@]}"; do
    echo "*****Running experiment with num_outputs=${num_outputs} and model_path=${model_path}*****"
    python predict_logits_copy.py $num_outputs $model_path
done