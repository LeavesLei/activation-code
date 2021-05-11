#!/bin/bash

# SBATCH -A test
#SBATCH -J adult_ss
#SBATCH -p nips
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -o adult_ss.out
#SBATCH -t 5:00:00

cd /public/data1/users/leishiye
source .bashrc
conda activate tf-gpu
cd neural_code/activation-code/Adult

python run_adult_sample_size_encoding_properties.py