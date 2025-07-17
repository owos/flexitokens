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

exp_dir=~/owos/experiments/fxt/model_ckpts/
C=configs/train/fxt_baseline_1_bp_6_priors_0.3_en_hard_no_binomial_lambda_3_layers3_12_3.yaml
GPUS=2
accelerate_config_file=configs/accelerate/gpu_$GPUS.yaml
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo 'Run training...'

if [ -z $GPUS ]
then
    python src/train.py --config_file "$C" --work_dir $work_dir 
else
    echo 'Finding free port'    
    PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    accelerate launch --main_process_port=$PORT --config_file=$accelerate_config_file --num_processes="$GPUS" src/train/train.py --config_file "$C" --exp_dir $exp_dir --with_tracking True 
    
fi

