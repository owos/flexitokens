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

C=configs/finetune/sib200_routing.yml
GPUS=1
config_file=configs/accelerate/gpu_1.yaml

models=(
    "model_ckpts/fxt_baseline_1_bp_6_priors_0.3_en_hard_no_binomial_lambda_3/_2025-05-03_14-43-31"

)

SEEDS=(42)
LRS=(5e-5)
BSZS=(8)
gradient_accumulation_steps=2
LANGS=(en es ru uk hi te)


for model in "${models[@]}"; do
    echo "Starting with model ${model}"
    work_dir="~/owos/experiments/fxt/model_ckpts/downstream"
    echo $work_dir
    for SEED in "${SEEDS[@]}";do
        echo "Starting with seed ${SEED}"

        for LR in "${LRS[@]}";do

            for BSZ in "${BSZS[@]}";do

                for language in "${LANGS[@]}"; do
                    echo 'Finding free port'
                    PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
                    accelerate launch --main_process_port=$PORT --config_file=$config_file --num_processes="$GPUS" src/finetune/train_classification.py  \
                        --config_file "$C" \
                        --pretrained_path "model_ckpts/${model}/" \
                        --work_dir $work_dir \
                        --language $language \
                        --lr  $LR \
                        --batch_size $BSZ \
                        --seed $SEED \
                        --gradient_accumulation_steps $gradient_accumulation_steps \
                        --use_best_model False \

                done
            done
        done
    done
done