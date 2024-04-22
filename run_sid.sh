#!/bin/bash

dataset=$1

# Uncomment to set CUDA visible devices and run specific script

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# sh run_sid.sh 'cifar10-uncond' 

# Modify --duration to reproduce the reported results

if [ "$dataset" = 'cifar10-uncond' ]; then
    # Command to run torch with specific parameters
    # Add the option below to load a checkpoint:
    # --resume 'image_experiment/sid-train-runs/cifar10-uncond/training-state-????.pt'
    torchrun --standalone --nproc_per_node=4 sid_train.py \
    --alpha 1.2 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 256 \
    --batch-gpu 16 \
    --outdir 'image_experiment/sid-train-runs/cifar10-uncond' \
    --data 'image_experiment/datasets/cifar10-32x32.zip' \
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
    --duration 100

elif [ "$dataset" = 'cifar10-cond' ]; then
    torchrun --standalone --nproc_per_node=4 sid_train.py \
    --alpha 1.2 \
    --cond 1 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 256 \
    --batch-gpu 16 \
    --outdir 'image_experiment/sid-train-runs/cifar10-cond' \
    --data 'image_experiment/datasets/cifar10-32x32.zip' \
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
    --duration 100
    
elif [ "$dataset" = 'imagenet64-cond' ]; then
    torchrun --standalone --nproc_per_node=4 sid_train.py \
    --alpha 1.2 \
    --cond 1 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 8192 \
    --batch-gpu 16 \
    --outdir 'image_experiment/sid-train-runs/imagenet64-cond' \
    --data 'image_experiment/datasets/imagenet-64x64.zip' \
    --arch adm \
    --edm_model imagenet64-cond \
    --metrics fid50k_full \
    --tick 10 \
    --snap 50 \
    --dump 500 \
    --lr 5e-6 \
    --glr 5e-6 \
    --fp16 1 \
    --ls 1 \
    --lsg 100 \
    --tick 20 \ 
    --snap 50 \
    --dump 500 \
    --dropout 0.1 \
    --augment 0 \
    --ema 2 \
    --duration 100 
    
elif [ "$dataset" = 'ffhq64' ]; then
    torchrun --standalone --nproc_per_node=4 sid_train.py \
    --alpha 1.0 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 512 \
    --batch-gpu 32 \
    --outdir 'image_experiment/sid-train-runs/ffhq64' \
    --data 'image_experiment/datasets/ffhq-64x64.zip' \
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
    --tick 10 \ 
    --snap 50 \
    --dump 500 \
    --dropout 0.05 \
    --augment 0.15 \
    --cres 1,2,2,2 \
    --g_beta1 0.9 \
    --duration 100 
           
elif [ "$dataset" = 'afhq64-v2' ]; then
    torchrun --standalone --nproc_per_node=4 sid_train.py \
    --alpha 1.2 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 512 \
    --batch-gpu 32 \
    --outdir 'image_experiment/sid-train-runs/afhq64-v2' \
    --data 'image_experiment/datasets/afhqv2-64x64.zip' \
    --arch ddpmpp \
    --edm_model imagenet64-cond \
    --metrics fid50k_full \
    --tick 10 \
    --snap 50 \
    --dump 500 \
    --lr 5e-6 \
    --glr 5e-6 \
    --fp16 1 \
    --ls 1 \
    --lsg 100 \
    --tick 10 \ 
    --snap 50 \
    --dump 500 \
    --dropout 0.05 \
    --augment 0.15 \
    --cres 1,2,2,2 \
    --duration 100 
else
    echo "Invalid dataset specified"
    exit 1
fi
           
            
        
          