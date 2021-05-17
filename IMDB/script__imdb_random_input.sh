#!/bin/bash

# SBATCH -A test
#SBATCH -J imdb_random
#SBATCH -p nips
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -o imdb_random.out
#SBATCH -t 5:00:00

cd /public/data1/users/leishiye
source .bashrc
conda activate tf-gpu
cd neural_code/activation-code/IMDB

python run_imdb_training_time_random_input.py