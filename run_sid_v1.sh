#!/bin/bash

# Reproduce SiD distillation of pretrained EDM models
# Enhanced version of run_sid.sh with additional features:
# - Automatically resumes training from the latest checkpoint
# - Saves only the best model (based on FID) and the latest checkpoint

# Retrieve the dataset name from the first argument
dataset=$1

# Example usage:
# To set specific GPUs and run the script for 'cifar10-uncond':
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# sh run_sid_sida.sh 'cifar10-uncond'

# Tip: Decrease --batch-gpu to reduce memory consumption on limited GPU resources
#   Decrease --duration to enable earlier termination, as FID improvement slows significantly 
#   at later stages due to the log-log linear relationship between FID and the number of iterations.

if [ "$dataset" = 'cifar10-uncond' ]; then
    # Command to execute the SiD training script with specified parameters
    # Optional: Use the --resume option to load a specific checkpoint, e.g.:
    # --resume 'image_experiment/sid-train-runs/cifar10-uncond/training-state-????.pt'
    # If --resume points to a folder, the script will automatically load the latest checkpoint from that folder. 
    # This is particularly useful for seamless resumption when running the code in a cluster environment.
    # Note: Optional parameters, such as --data_stat, will be computed automatically within the code if not explicitly provided.
    torchrun --standalone --nproc_per_node=4 sida_train.py \
    --alpha 1.2 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 256 \
    --batch-gpu 32 \
    --data '/data/datasets/cifar10-32x32.zip'  \
    --outdir '/data/image_experiment/sid-train-runs/cifar10-uncond' \
    --resume '/data/image_experiment/sid-train-runs/cifar10-uncond' \
    --nosubdir 0 \
    --arch ddpmpp \
    --edm_model cifar10-uncond \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --tick 10 \
    --snap 50 \
    --dump 200 \
    --lr 1e-5 \
    --glr 1e-5 \
    --fp16 0 \
    --ls 1 \
    --lsg 100 \
    --duration 500 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz' \
    --use_gan 0 \
    --metrics fid50k_full,is50k \
    --save_best_and_last 1 

    
elif [ "$dataset" = 'cifar10-cond' ]; then
    torchrun --standalone --nproc_per_node=4 sida_train.py \
    --cond 1 \
    --alpha 1.2 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 256 \
    --batch-gpu 32 \
    --data '/data/datasets/cifar10-32x32.zip'  \
    --outdir '/data/image_experiment/sid-train-runs/cifar10-cond' \
    --resume '/data/image_experiment/sid-train-runs/cifar10-cond' \
    --nosubdir 0 \
    --arch ddpmpp \
    --edm_model cifar10-cond \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --tick 10 \
    --snap 50 \
    --dump 200 \
    --lr 1e-5 \
    --glr 1e-5 \
    --fp16 0 \
    --ls 1 \
    --lsg 100 \
    --duration 800 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz' \
    --use_gan 0 \
    --metrics fid50k_full \
    --save_best_and_last 1 
    
    
elif [ "$dataset" = 'imagenet64-cond' ]; then
    torchrun --standalone --nproc_per_node=4 sida_train.py \
    --cond 1.2 \
    --alpha 1 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 8192 \
    --batch-gpu 32 \
    --data '/data/datasets/imagenet-64x64.zip' \
    --outdir '/data/image_experiment/sid-train-runs/imagenet64-cond' \
    --resume '/data/image_experiment/sid-train-runs/imagenet64-cond' \
    --nosubdir 0 \
    --arch adm \
    --edm_model imagenet64-cond \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --tick 20 \
    --snap 50 \
    --dump 200 \
    --lr 4e-6 \
    --glr 4e-6 \
    --fp16 1 \
    --ls 1 \
    --lsg 100 \
    --duration 1000 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz' \
    --use_gan 1 \
    --metrics fid50k_full \
    --save_best_and_last 1 \
    --dropout 0.1 \
    --augment 0 \
    --ema 2 \
    --duration 300 
    
    
elif [ "$dataset" = 'ffhq64' ]; then
    torchrun --standalone --nproc_per_node=4 sida_train.py \
    --alpha 1 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 512 \
    --batch-gpu 64 \
    --data '/data/datasets/ffhq-64x64.zip' \
    --outdir '/data/image_experiment/sid-train-runs/ffhq64' \
    --resume '/data/image_experiment/sid-train-runs/ffhq64' \
    --nosubdir 0 \
    --arch ddpmpp \
    --edm_model ffhq64 \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --tick 10 \
    --snap 50 \
    --dump 200 \
    --lr 1e-5 \
    --glr 1e-5 \
    --fp16 1 \
    --ls 1 \
    --lsg 100 \
    --duration 300 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz' \
    --use_gan 0 \
    --metrics fid50k_full \
    --save_best_and_last 1 \
    --dropout 0.05 \
    --augment 0.15 \
    --cres 1,2,2,2 \
    --duration 500 \
    --g_beta1 0.9 
    
           
elif [ "$dataset" = 'afhq64-v2' ]; then
    torchrun --standalone --nproc_per_node=4 sida_train.py \
    --alpha 1 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 512 \
    --batch-gpu 64 \
    --data '/data/datasets/ffhq-64x64.zip' \
    --outdir '/data/image_experiment/sid-train-runs/afhq64-v2' \
    --resume '/data/image_experiment/sid-train-runs/afhq64-v2' \
    --nosubdir 0 \
    --arch ddpmpp \
    --edm_model afhq64-v2 \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --tick 10 \
    --snap 50 \
    --dump 200 \
    --lr 5e-6 \
    --glr 5e-6 \
    --fp16 1 \
    --ls 1 \
    --lsg 100 \
    --lsd 1 \
    --lsg_gan 0.01 \
    --duration 200 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz' \
    --use_gan 0 \
    --metrics fid50k_full \
    --save_best_and_last 1 \
    --dropout 0.05 \
    --augment 0.15 \
    --cres 1,2,2,2 \
    --duration 500 
    
else
    echo "Invalid dataset specified"
    exit 1
fi
           
            
        
          
