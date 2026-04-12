#!/bin/bash
#SBATCH --job-name=difgan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

mkdir -p logs

cd /home/lab/lhoffmann/Diffusion-StyleGAN-tiny-n/diffusion-stylegan2

source /home/lab/lhoffmann/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-gan2

python3 train.py \
  --outdir=results/ST_3_PC \
  --data=/tmp/george_leilani \
  --gpus=1 --kimg=1000 \
  --snap=25

python3 train.py \
  --outdir=results/FACES \
  --data=/tmp/george_leilani/faces_multisex_128x128 \
  --gpus=1 --kimg=1000 \
  --snap=25
