#!/bin/bash

#SBATCH -A test
#SBATCH -J resnet_im
#SBATCH -p nips
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -o resnet_im.out
#SBATCH -t 20:00:00

cd /public/data1/users/leishiye
source .bashrc
cd neural_code/activation-code/ImageNet

python imagenet_resnet50.py