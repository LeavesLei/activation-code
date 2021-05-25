#!/bin/bash

#SBATCH -A test
#SBATCH -J resnext_im
#SBATCH -p nips
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -o resnext_im.out
#SBATCH -t 20:00:00

cd /public/data1/users/leishiye
source .bashrc
cd neural_code/activation-code/ImageNet

python imagenet_resnext50.py