#!/bin/bash

# SBATCH -A test
#SBATCH -J tt_ep_5
#SBATCH -p nips
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -o tt_ep_5.out
#SBATCH -t 14:00:00

cd /public/data1/users/leishiye
source .bashrc
cd neural_code/activation-code/TinyImagenet/sample_size_appendix

python run_tinyimagenet_training_time_encoding_properties.py --begin_repeat 5