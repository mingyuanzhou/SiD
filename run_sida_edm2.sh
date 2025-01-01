#!/bin/bash

dataset=$1

# Run the code below in command window to set CUDA visible devices and run specific script
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#sh run_sida_edm2.sh 'imagenet512-xs'

# Decrease --batch-gpu to reduce memory consumption


if [ "$dataset" = 'test_run' ]; then
    # Command to run torch with specific parameters
    # Many options are optional, such as --data_stat, which will be computed inside the code if not provided
    # Add the option below to load a checkpoint:
    # --resume 'image_experiment/sid-train-runs/cifar10-uncond/training-state-????.pt'

    #torchrun --standalone --nproc_per_node=4 sida_edm2_train.py \
    python -m torch.distributed.run --nproc_per_node=3 sida_edm2_train.py \
    --use_gan 1 \
    --force_normalization 1 \
    --return_logvar 0 \
    --cond 1 \
    --alpha 1 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 2400 \
    --batch-gpu 32 \
    --outdir '/data/image_experiment/sid-train-runs/edm2-xs' \
    --resume '/data/image_experiment/sid-train-runs/edm2-xs' \
    --nosubdir 1 \
    --precond 'edm2' \
    --arch 'edm2-img512-xs' \
    --edm_model 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-2147483-0.135.pkl' \
    --tick 10 \
    --snap 10 \
    --dump 200 \
    --lr 5e-5 \
    --glr 5e-5 \
    --fp16 1 \
    --ls 1 \
    --lsg 100 \
    --duration 200 \
    --augment 0 \
    --ema 2 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl' \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --metrics fid_test \
    --data '/data/datasets/img512-sd.zip' \
    --lsd 100 \
    --lsg_gan 0.01
    #--vae '/data/datasets/stabilityai/sd-vae-ft-mse' \
    #--metrics fid_test \
    #--resume '/data/image_experiment/sid-train-runs/edm2-xs' \
    
elif [ "$dataset" = 'imagenet512-xs' ]; then
    # Command to run torch with specific parameters
    # Many options are optional, such as --data_stat, which will be computed inside the code if not provided
    # Add the option below to load a checkpoint:
    # --resume 'image_experiment/sid-train-runs/cifar10-uncond/training-state-????.pt'

    #torchrun --standalone --nproc_per_node=4 sida_edm2_train.py \
    python -m torch.distributed.run --nproc_per_node=3 sida_edm2_train.py \
    --use_gan 1 \
    --force_normalization 1 \
    --return_logvar 1 \
    --cond 1 \
    --alpha 1 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 2048 \
    --batch-gpu 64 \
    --outdir '/data/image_experiment/sida-train-runs/edm2-xs' \
    --resume '/data/image_experiment/sida-train-runs/edm2-xs' \
    --nosubdir 0 \
    --precond 'edm2' \
    --arch 'edm2-img512-xs' \
    --edm_model 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-2147483-0.135.pkl' \
    --tick 10 \
    --snap 50 \
    --dump 200 \
    --lr 5e-5 \
    --glr 5e-5 \
    --fp16 1 \
    --ls 1 \
    --lsg 100 \
    --duration 200 \
    --augment 0 \
    --ema 2 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl' \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --metrics fid50k_full \
    --data '/data/datasets/img512-sd.zip' \
    --lsd 100 \
    --lsg_gan 0.01
    #--vae '/data/datasets/stabilityai/sd-vae-ft-mse' \ #vae can be saved into this folder
    #--metrics fid_test \ #use this option to compute FID using 512 images instead of 50000 images

elif [ "$dataset" = 'imagenet512-s' ]; then

    python -m torch.distributed.run --nproc_per_node=4 sida_edm2_train.py \
    --use_gan 1 \
    --force_normalization 0 \
    --return_logvar 0 \
    --cond 1 \
    --alpha 1 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 2048 \
    --batch-gpu 64 \
    --outdir '/data/image_experiment/sida-train-runs/edm2-s' \
    --resume '/data/image_experiment/sida-train-runs/edm2-s' \
    --nosubdir 0 \
    --precond 'edm2' \
    --arch 'edm2-img512-s' \
    --edm_model 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-s-2147483-0.130.pkl' \
    --tick 10 \
    --snap 50 \
    --dump 200 \
    --lr 5e-5 \
    --glr 5e-5 \
    --fp16 1 \
    --ls 1 \
    --lsg 100 \
    --duration 200 \
    --augment 0 \
    --ema 2 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl' \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --metrics fid50k_full \
    --data '/data/datasets/img512-sd.zip' \
    --lsd 100 \
    --lsg_gan 0.01

elif [ "$dataset" = 'imagenet512-m' ]; then

    python -m torch.distributed.run --nproc_per_node=4 sida_edm2_train.py \
    --use_gan 1 \
    --force_normalization 1 \
    --return_logvar 1 \
    --cond 1 \
    --alpha 1 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 2048 \
    --batch-gpu 32 \
    --outdir '/data/image_experiment/sida-train-runs/edm2-m' \
    --resume '/data/image_experiment/sida-train-runs/edm2-m' \
    --nosubdir 0 \
    --precond 'edm2' \
    --arch 'edm2-img512-m' \
    --edm_model 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-m-2147483-0.100.pkl' \
    --tick 10 \
    --snap 50 \
    --dump 200 \
    --lr 5e-5 \
    --glr 5e-5 \
    --fp16 1 \
    --ls 1 \
    --lsg 100 \
    --duration 200 \
    --augment 0 \
    --ema 2 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl' \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --metrics fid50k_full \
    --data '/data/datasets/img512-sd.zip' \
    --lsd 100 \
    --lsg_gan 0.01
    
elif [ "$dataset" = 'imagenet512-l' ]; then

    python -m torch.distributed.run --nproc_per_node=4 sida_edm2_train.py \
    --use_gan 1 \
    --force_normalization 1 \
    --return_logvar 1 \
    --cond 1 \
    --alpha 1 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 2048 \
    --batch-gpu 32 \
    --outdir '/data/image_experiment/sida-train-runs/edm2-l' \
    --resume '/data/image_experiment/sida-train-runs/edm2-l' \
    --nosubdir 0 \
    --precond 'edm2' \
    --arch 'edm2-img512-l' \
    --edm_model 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-l-1879048-0.085.pkl' \
    --tick 10 \
    --snap 50 \
    --dump 200 \
    --lr 5e-5 \
    --glr 5e-5 \
    --fp16 1 \
    --ls 1 \
    --lsg 100 \
    --duration 200 \
    --augment 0 \
    --ema 2 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl' \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --metrics fid50k_full \
    --data '/data/datasets/img512-sd.zip' \
    --lsd 100 \
    --lsg_gan 0.01
    
elif [ "$dataset" = 'imagenet512-xl' ]; then

    python -m torch.distributed.run --nproc_per_node=4 sida_edm2_train.py \
    --use_gan 1 \
    --force_normalization 1 \
    --return_logvar 1 \
    --cond 1 \
    --alpha 1 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 2048 \
    --batch-gpu 16 \
    --outdir '/data/image_experiment/sida-train-runs/edm2-xl' \
    --resume '/data/image_experiment/sida-train-runs/edm2-xl' \
    --nosubdir 0 \
    --precond 'edm2' \
    --arch 'edm2-img512-xl' \
    --edm_model 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xl-1342177-0.085.pkl' \
    --tick 10 \
    --snap 50 \
    --dump 200 \
    --lr 5e-5 \
    --glr 5e-5 \
    --fp16 1 \
    --ls 1 \
    --lsg 100 \
    --duration 200 \
    --augment 0 \
    --ema 2 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl' \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --metrics fid50k_full \
    --data '/data/datasets/img512-sd.zip' \
    --lsd 100 \
    --lsg_gan 0.01

elif [ "$dataset" = 'imagenet512-xxl' ]; then

    python -m torch.distributed.run --nproc_per_node=4 sida_edm2_train.py \
    --use_gan 1 \
    --force_normalization 1 \
    --return_logvar 1 \
    --cond 1 \
    --alpha 1 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 2048 \
    --batch-gpu 2 \
    --outdir '/data/image_experiment/sida-train-runs/edm2-xxl' \
    --resume '/data/image_experiment/sida-train-runs/edm2-xxl' \
    --nosubdir 0 \
    --precond 'edm2' \
    --arch 'edm2-img512-xxl' \
    --edm_model 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xxl-0939524-0.070.pkl' \
    --tick 10 \
    --snap 50 \
    --dump 200 \
    --lr 5e-5 \
    --glr 5e-5 \
    --fp16 1 \
    --ls 1 \
    --lsg 100 \
    --duration 100 \
    --augment 0 \
    --ema 2 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl' \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --metrics fid50k_full \
    --data '/data/datasets/img512-sd.zip' \
    --lsd 100 \
    --lsg_gan 0.01
    
else
    echo "Invalid dataset specified"
    exit 1
fi