#!/bin/bash

# SBATCH -A test
#SBATCH -J ti_ss_5
#SBATCH -p nips
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -o ti_ss_5.out
#SBATCH -t 10:00:00

cd /public/data1/users/leishiye
source bashrc-bak
cd neural_code/activation-code/TinyImagenet

python run_tinyimagenet_sample_size_train_cnn.py --begin_repeat 5