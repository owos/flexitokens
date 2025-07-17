#!/bin/bash
#SBATCH --account ...
#SBATCH --job-name hin40
# ...



source ~/miniforge3/etc/profile.d/conda.sh
conda activate fxt
conda env list 
nvidia-smi

cat $0
echo "--------------------"


export PYTHONPATH=$(pwd)
export HF_HOME="cache"
export WANDB_CACHE_DIR="cache"

GPUS=1
config_file=configs/accelerate/gpu_1.yaml


models=(
    "model_ckpts/fxt_baseline_1_bp_6_priors_0.3_en_hard_no_binomial_lambda_3/_2025-05-03_14-43-31"

)


for model in "${models[@]}"; do
    echo "Starting with model ${model}"
    work_dir="model_ckpts/${model}/"
    echo $work_dir
    echo 'Finding free port'
    PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    accelerate launch --main_process_port=$PORT --config_file=$config_file --num_processes="$GPUS" src/eval/compression_rate.py  \
      --task "flores" \
      --model_path $work_dir \
      --output_dir "results/compression_rate/" \
      --batch_size 1 
done