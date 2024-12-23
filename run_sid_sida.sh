#!/bin/bash

#Reproduce SiD-SiDA (SiD^2A) distillation of pretrained EDM models

# Retrieve the dataset name from the first argument
dataset=$1

# Example usage:
# To set specific GPUs and run the script for 'cifar10-uncond':
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# sh run_sida.sh 'cifar10-uncond'

# Tip: Decrease --batch-gpu to reduce memory consumption on limited GPU resources

if [ "$dataset" = 'cifar10-uncond' ]; then
    # Command to execute the SiDA training script with specified parameters
    # Optional: Use the --resume option to load a specific checkpoint, e.g.:
    # --resume 'image_experiment/sid-train-runs/cifar10-uncond/training-state-????.pt'
    # If --resume points to a folder, the script will automatically load the latest checkpoint from that folder. 
    # This is particularly useful for seamless resumption when running the code in a cluster environment.
    # Note: Optional parameters, such as --data_stat, will be computed automatically within the code if not explicitly provided.
    torchrun --standalone --nproc_per_node=4 sida_train.py \
    --alpha 1 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 256 \
    --batch-gpu 32 \
    --data '/data/datasets/cifar10-32x32.zip'  \
    --outdir '/data/image_experiment/sida-train-runs/cifar10-uncond' \
    --resume '/data/image_experiment/sida-train-runs/cifar10-uncond' \
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
    --lsd 1 \
    --lsg_gan 0.01 \
    --duration 100 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz' \
    --use_gan 1 \
    --metrics fid50k_full,is50k \
    --save_best_and_last 1 \
    --sid_model 'https://huggingface.co/UT-Austin-PML/SiD/resolve/main/cifar10-uncond/alpha1.2/network-snapshot-1.200000-403968.pkl'
    
    # torchrun --standalone --nproc_per_node=4 sida_train.py \
    # --alpha 1 \                            # Scaling factor for gradient-bias correction in SiD
    # --tmax 800 \                           # Maximum diffusion time steps considered for distillation
    # --init_sigma 2.5 \                     # Input noise standard devivation of the generator
    # --batch 256 \                          # Total batch size across all GPUs
    # --batch-gpu 32 \                       # Per-GPU batch size
    # --outdir '/data/image_experiment/sida-train-runs/cifar10-uncond' \ # Output directory for training results
    # --resume '/data/image_experiment/sida-train-runs/cifar10-uncond' \ # Checkpoint folder for resuming training
    # --nosubdir 0 \                         # Flag to control subdirectory creation; a subfolder will be created based on the parameter values when set to 0   
    # --data '/data/datasets/cifar10-32x32.zip'  \  # Training dataset (SiDA requires images; not data-free)
    # --arch ddpmpp \                        # Model architecture
    # --edm_model cifar10-uncond \           # Pretrained model for comparison or initialization
    # --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \ # Inception model for FID computation
    # --tick 10 \                            # Frequency (in k images) for status updates
    # --snap 50 \                            # Frequency (in tick) for saving G_ema (.pkl file)
    # --dump 500 \                           # Frequency (in tick) for saving checkpoint (.pt file)
    # --lr 1e-5 \                            # Learning rate for the fake score network
    # --glr 1e-5 \                           # Learning rate for the generator
    # --fp16 0 \                             # Flag for mixed-precision training (0 = disable)
    # --ls 1 \                               # Scaling factor for fake score network loss
    # --lsg 100 \                            # Scaling factor for generator loss
    # --lsd 1 \                              # Discriminator loss weight
    # --lsg_gan 0.01 \                       # GAN loss weight
    # --duration 300 \                       # Total training duration (in millons of images)
    # --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz' \ # Dataset statistics for FID calculation (optional if --data is specified)    
    # --use_gan 1 \                          # Enable GAN loss
    # --metrics fid50k_full,is50k \          # Metrics for evaluation (FID and Inception Score)
    # --save_best_and_last 1                 # Only save G_ema with the best FID and the latest checkpoing for resuming
    # # Optional: Specify a pretrained SiDA model using --sid_model, e.g.:
    # --sid_model 'https://huggingface.co/UT-Austin-PML/SiD/resolve/main/cifar10-uncond/alpha1.2/network-snapshot-1.200000-403968.pkl'

    
elif [ "$dataset" = 'cifar10-cond' ]; then
    torchrun --standalone --nproc_per_node=4 sida_train.py \
    --cond 1 \
    --alpha 1 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 256 \
    --batch-gpu 32 \
    --data '/data/datasets/cifar10-32x32.zip'  \
    --outdir '/data/image_experiment/sida-train-runs/cifar10-cond' \
    --resume '/data/image_experiment/sida-train-runs/cifar10-cond' \
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
    --lsd 1 \
    --lsg_gan 0.01 \
    --duration 100 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz' \
    --use_gan 1 \
    --metrics fid50k_full \
    --save_best_and_last 1 \
    --sid_model 'https://huggingface.co/UT-Austin-PML/SiD/resolve/main/cifar10-cond/alpha1.2/network-snapshot-1.200000-713312.pkl'

    
elif [ "$dataset" = 'imagenet64-cond' ]; then
    torchrun --standalone --nproc_per_node=4 sida_train.py \
    --cond 1 \
    --alpha 1 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 8192 \
    --batch-gpu 32 \
    --data '/data/datasets/imagenet-64x64.zip' \
    --outdir '/data/image_experiment/sida-train-runs/imagenet64-cond' \
    --resume '/data/image_experiment/sida-train-runs/imagenet64-cond' \
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
    --lsd 1 \
    --lsg_gan 0.01 \
    --duration 300 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz' \
    --use_gan 1 \
    --metrics fid50k_full \
    --save_best_and_last 1 \
    --dropout 0.1 \
    --augment 0 \
    --ema 2 \
    --duration 100 \
    --sid_model 'https://huggingface.co/UT-Austin-PML/SiD/resolve/main/imagenet64/alpha1.2/network-snapshot-1.200000-939176.pkl'
    
elif [ "$dataset" = 'ffhq64' ]; then
    torchrun --standalone --nproc_per_node=4 sida_train.py \
    --alpha 1 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 512 \
    --batch-gpu 64 \
    --data '/data/datasets/ffhq-64x64.zip' \
    --outdir '/data/image_experiment/sida-train-runs/ffhq64' \
    --resume '/data/image_experiment/sida-train-runs/ffhq64' \
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
    --lsd 1 \
    --lsg_gan 0.01 \
    --duration 300 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz' \
    --use_gan 1 \
    --metrics fid50k_full \
    --save_best_and_last 1 \
    --dropout 0.05 \
    --augment 0.15 \
    --cres 1,2,2,2 \
    --duration 100 \
    --sid_model 'https://huggingface.co/UT-Austin-PML/SiD/resolve/main/ffhq64/alpha1.2/network-snapshot-1.200000-498176.pkl'

           
elif [ "$dataset" = 'afhq64-v2' ]; then
    torchrun --standalone --nproc_per_node=4 sida_train.py \
    --alpha 1 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 512 \
    --batch-gpu 64 \
    --data '/data/datasets/ffhq-64x64.zip' \
    --outdir '/data/image_experiment/sida-train-runs/afhq64-v2' \
    --resume '/data/image_experiment/sida-train-runs/afhq64-v2' \
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
    --duration 300 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz' \
    --use_gan 1 \
    --metrics fid50k_full \
    --save_best_and_last 1 \
    --dropout 0.05 \
    --augment 0.15 \
    --cres 1,2,2,2 \
    --duration 100 \
    --sid_model 'https://huggingface.co/UT-Austin-PML/SiD/resolve/main/afhq64/alpha1/network-snapshot-1.000000-371712.pkl'
    
else
    echo "Invalid dataset specified"
    exit 1
fi
           
            
        
          
