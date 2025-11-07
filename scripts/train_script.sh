#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU
#SBATCH -t 00:40:00
#SBATCH --gpus=h100-80:8
#echo commands to stdout
set -x

cd $SLURM_SUBMIT_DIR

module load python/3.8.6
module load cuda/12.6.1

VENV_PATH=~/venvs/foundation_privacy

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment..."
    python -m venv $VENV_PATH
    source $VENV_PATH/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "Using existing virtual environment..."
    source $VENV_PATH/bin/activate
fi

python ft_llm.py \
  --data_dir ../shadow_data_canary/shadow_4 --train_file "train_finetune.json" \
  -m gpt2 --block_size 512 --epochs 3 --batch_size 8 --gradient_accumulation_steps 1 \
  --lr 2e-4 --outdir ../models/shadow_4/gpt2_shadow \
  --lora --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --merge_lora



