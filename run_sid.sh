#!/bin/bash

dataset=$1

# Run the code below in command window to set CUDA visible devices and run specific script
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#sh run_sid.sh 'cifar10-uncond' 

# Modify --duration to reproduce the reported results
# Decrease --batch-gpu to reduce memory consumption

if [ "$dataset" = 'cifar10-uncond' ]; then
    # Command to run torch with specific parameters
    # Add the option below to load a checkpoint:
    #Many options are optional, such as --data_stat, which will be computed inside the code if not provided
    # --resume 'image_experiment/sid-train-runs/cifar10-uncond/training-state-????.pt'
    torchrun --standalone --nproc_per_node=4 sid_train.py \
    --alpha 1.2 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 256 \
    --batch-gpu 16 \
    --outdir 'image_experiment/sid-train-runs/cifar10-uncond' \
    --data '/data/datasets/cifar10-32x32.zip' \
    --arch ddpmpp \
    --edm_model cifar10-uncond \
    --metrics fid50k_full,is50k \
    --tick 10 \
    --snap 50 \
    --dump 500 \
    --lr 1e-5 \
    --glr 1e-5 \
    --fp16 0 \
    --ls 1 \
    --lsg 100 \
    --duration 100 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz'
    #\
    #--resume 'image_experiment/sid-train-runs/cifar10-uncond/00002-cifar10-32x32-uncond-ddpmpp-edmglr1e-05-lr1e-05-initsigma2.5-gpus4-alpha1.2-batch256-tmax800-fp32/training-state-000128.pt'

elif [ "$dataset" = 'cifar10-cond' ]; then
    torchrun --standalone --nproc_per_node=4 sid_train.py \
    --alpha 1.2 \
    --cond 1 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 256 \
    --batch-gpu 16 \
    --outdir 'image_experiment/sid-train-runs/cifar10-cond' \
    --data '/data/datasets/cifar10-32x32.zip' \
    --arch ddpmpp \
    --edm_model cifar10-cond \
    --metrics fid50k_full \
    --tick 10 \
    --snap 50 \
    --dump 500 \
    --lr 1e-5 \
    --glr 1e-5 \
    --fp16 0 \
    --ls 1 \
    --lsg 100 \
    --duration 100 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz'
    
elif [ "$dataset" = 'imagenet64-cond' ]; then
    torchrun --standalone --nproc_per_node=4 sid_train.py \
    --alpha 1.2 \
    --cond 1 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 8192 \
    --batch-gpu 16 \
    --outdir 'image_experiment/sid-train-runs/imagenet64-cond' \
    --data '/data/datasets/imagenet-64x64.zip' \
    --arch adm \
    --edm_model imagenet64-cond \
    --metrics fid50k_full \
    --tick 20 \
    --snap 50 \
    --dump 500 \
    --lr 4e-6 \
    --glr 4e-6 \
    --fp16 1 \
    --ls 1 \
    --lsg 100 \
    --dropout 0.1 \
    --augment 0 \
    --ema 2 \
    --duration 100 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz'
    
elif [ "$dataset" = 'ffhq64' ]; then
    torchrun --standalone --nproc_per_node=4 sid_train.py \
    --alpha 1.2 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 512 \
    --batch-gpu 32 \
    --outdir 'image_experiment/sid-train-runs/ffhq64' \
    --data '/data/datasets/ffhq-64x64.zip' \
    --arch ddpmpp \
    --edm_model ffhq64 \
    --metrics fid50k_full \
    --tick 10 \
    --snap 50 \
    --dump 500 \
    --lr 1e-5 \
    --glr 1e-5 \
    --fp16 1 \
    --ls 1 \
    --lsg 100 \
    --dropout 0.05 \
    --augment 0.15 \
    --cres 1,2,2,2 \
    --g_beta1 0.9 \
    --duration 100 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz'
           
elif [ "$dataset" = 'afhq64-v2' ]; then
    torchrun --standalone --nproc_per_node=4 sid_train.py \
    --alpha 1.0 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 512 \
    --batch-gpu 32 \
    --outdir 'image_experiment/sid-train-runs/afhq64-v2' \
    --data '/data/datasets/afhqv2-64x64.zip' \
    --arch ddpmpp \
    --edm_model afhq64-v2 \
    --metrics fid50k_full \
    --tick 10 \
    --snap 50 \
    --dump 500 \
    --lr 5e-6 \
    --glr 5e-6 \
    --fp16 1 \
    --ls 1 \
    --lsg 100 \
    --dropout 0.05 \
    --augment 0.15 \
    --cres 1,2,2,2 \
    --duration 100 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz'
else
    echo "Invalid dataset specified"
    exit 1
fi
           
            
        
          
