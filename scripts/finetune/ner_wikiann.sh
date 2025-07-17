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

C=configs/finetune/ner_wikiann.yml
GPUS=1
config_file=configs/accelerate/gpu_1.yaml

models=(
    # fxt_baseline_1_bp_6_priors_0.1_en_hard_no_binomial/_2025-04-29_22-00-31
    # fxt_baseline_1_bp_6_priors_0.2_en_hard_no_binomial/_2025-04-29_17-06-35
    # fxt_baseline_1_bp_6_priors_0.3_en_hard_no_binomial/_2025-04-29_16-29-47
    # fxt_baseline_1_bp_6_priors_0.3_en_hard_yes_binomial/_2025-04-30_06-11-02
    # fxt_baseline_1_bp_6_priors_0.3_en_hard_no_binomial_lambda_1/_2025-05-04_09-31-46
    # fxt_baseline_1_bp_6_priors_0.3_en_hard_no_binomial_lambda_3/_2025-05-03_14-43-31
    # fxt_baseline_1_bp_6_priors_BPE_50k/_2025-05-10_23-51-40
    # fxt_baseline_1_bp_6_priors_BPE/_2025-05-09_20-40-49
    # fxt_baseline_1_bp_6_priors_0.3_en_hard_no_binomial_lambda_1_scaled_std/2025-05-21_00-59-56_2025-05-21_00-59-56
    # fxt_baseline_1_bp_6_priors_0.3_en_hard_no_binomial_lambda_3_scaled_std/2025-05-21_00-46-22_2025-05-21_00-46-22
    # fxt_baseline_1_bp_6_priors_0.1_en_hard_no_binomial_lower3_scaled_std/2025-05-21_00-52-03_2025-05-21_00-52-03
    # fxt_baseline_1_bp_6_priors_0.3_en_hard_no_binomial_1B
    # fxt_baseline_1_bp_6_priors_0.3_en_hard_no_binomial_no_lower_bound
    # fxt_baseline_1_bp_6_priors_0.3_en_hard_no_binomial_1B_2e-4_bz1024_4gpus/step_32000
    # fxt_baseline_1_bp_6_priors_0.3_en_hard_no_binomial_1B_2e-4_bz512_4gpus/2025-06-13_13-31-15_2025-06-13_13-31-15
    fxt_baseline_1_bp_6_priors_0.3_en_hard_no_binomial_1B_2e-4_bz512_4gpus/2025-06-26_12-46-43_2025-06-26_12-46-43/step_50000


)

SEEDS=(42)
LRS=(5e-5)
BSZS=(1)
gradient_accumulation_steps=16
LANGS=(en)


for model in "${models[@]}"; do
    echo "Starting with model ${model}"
    # work_dir="model_ckpts/downstream/${model}/"
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