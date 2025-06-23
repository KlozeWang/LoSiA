for task in "piqa" "winogrande" "arc_c" "arc_e" "openbookqa" "siqa" 
do
    torchrun --standalone --nproc_per_node 1 $(dirname "$0")/../torchrun_main.py \
        --model_path meta-llama/Llama-2-7b-hf \
        --dataset_name ${task} \
        --dataset_path PATH_TO_TASK \
        --save_dir ${task}_losia \
        --lr 0.00005 \
        --batch_size 16 \
        --rank_factor 0.125 \
        --period 50 \
        --max_length 256 \
        --epochs 3 \
        --pad_to_max_len \
        --warmup_steps_ratio 0.1 \
        --grad_clipping 1.0 \
        --dtype bfloat16 \
        --single_gpu \
        --scheduler cosine_restarts \
        --optimizer losia_adamw_per_layer
done

for task in "arc_c" "boolq" 
do
    torchrun --standalone --nproc_per_node 1 $(dirname "$0")/../torchrun_main.py \
        --model_path meta-llama/Llama-2-7b-hf \
        --dataset_name ${task} \
        --dataset_path PATH_TO_TASK \
        --save_dir ${task}_losia \
        --lr 0.0002 \
        --batch_size 16 \
        --rank_factor 0.125 \
        --period 50 \
        --max_length 256 \
        --epochs 3 \
        --pad_to_max_len \
        --warmup_steps_ratio 0.1 \
        --grad_clipping 1.0 \
        --dtype bfloat16 \
        --single_gpu \
        --scheduler cosine_restarts \
        --optimizer losia_adamw_per_layer
done