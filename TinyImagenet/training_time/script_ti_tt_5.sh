#!/bin/bash

# SBATCH -A test
#SBATCH -J ti_tt_5
#SBATCH -p nips
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -o ti_tt_5.out
#SBATCH -t 10:00:00

cd /public/data1/users/leishiye
source .bashrc
cd neural_code/activation-code/TinyImagenet

python run_tinyimagenet_training_time_encoding_properties.py --begin_repeat 5