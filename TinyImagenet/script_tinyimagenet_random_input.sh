#!/bin/bash

# SBATCH -A test
#SBATCH -J tiny_random
#SBATCH -p nips
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -o tiny_random.out
#SBATCH -t 5:00:00

cd /public/data1/users/leishiye
source .bashrc
cd neural_code/activation-code/TinyImagenet/

python run_tinyimagenet_random_input.py --begin_repeat 1
python run_tinyimagenet_random_input.py --begin_repeat 2
python run_tinyimagenet_random_input.py --begin_repeat 3
python run_tinyimagenet_random_input.py --begin_repeat 4
python run_tinyimagenet_random_input.py --begin_repeat 5