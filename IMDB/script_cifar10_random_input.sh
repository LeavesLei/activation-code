#!/bin/bash

# SBATCH -A test
#SBATCH -J cifar10_random
#SBATCH -p nips
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -o cifar10_random.out
#SBATCH -t 5:00:00

cd /public/data1/users/leishiye
source .bashrc
conda activate tf-gpu
cd neural_code/activation-code

python run_cifar10_random_input.py