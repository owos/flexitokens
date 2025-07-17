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
export HF_HOME=~/owos/.cache/
export WANDB_CACHE_DIR=~/owos/.cache/

models=(
    "model_ckpts/fxt_baseline_1_bp_6_priors_0.3_en_hard_no_binomial_lambda_3/_2025-05-03_14-43-31"
)


for model in "${models[@]}"; do
    python  src/eval/evaluate_model.py \
        --model_path "$model" \
        --output_dir model_ckpts/results \
        --eval_batch_size 8
done

