#!/bin/bash

# SBATCH -A test
#SBATCH -J tt_t_4
#SBATCH -p nips
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -o tt_t_4.out
#SBATCH -t 10:00:00

cd /public/data1/users/leishiye
source .bashrc
cd neural_code/activation-code/SVHN

python run_svhn_training_time_training_mlp.py --begin_repeat 4