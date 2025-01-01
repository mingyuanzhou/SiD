#!/bin/bash

model=$1

# Run the code below in command window to set CUDA visible devices and run specific script
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# sh sid_metrics_edm2.sh 'imagenet512-xs-sida' 


# | Model    | Method  | Checkpoint URL                                                                                           |
# |----------|---------|---------------------------------------------------------------------------------------------------------|
# | EDM2-XS  | SiD     | [edm2_img512_xs_sid_alpha1-249360.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xs_sid_alpha1-249360.pkl) |
# | EDM2-XS  | SiDA    | [edm2_img512_xs_sida_alpha1-166410.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xs_sida_alpha1-166410.pkl) |
# | EDM2-XS  | SiD²A   | [edm2_img512_xs_sid2a_alpha1-028674.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xs_sid2a_alpha1-028674.pkl) |
# |----------|---------|---------------------------------------------------------------------------------------------------------|
# | EDM2-S   | SiD     | [edm2_img512_s_sid_alpha1-192010.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_s_sid_alpha1-192010.pkl) |
# | EDM2-S   | SiDA    | [edm2_img512_s_sida_alpha1-117764.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_s_sida_alpha1-117764.pkl) |
# | EDM2-S   | SiD²A   | [edm2_img512_s_sid2a_alpha1-075268.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_s_sid2a_alpha1-075268.pkl) |
# |----------|---------|---------------------------------------------------------------------------------------------------------|
# | EDM2-M   | SiD     | [edm2_img512_m_sid_alpha1-226830.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_m_sid_alpha1-226830.pkl) |
# | EDM2-M   | SiDA    | [edm2_img512_m_sida_alpha1-115716.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_m_sida_alpha1-115716.pkl) |
# | EDM2-M   | SiD²A   | [edm2_img512_m_sid2a_alpha1-044036.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_m_sid2a_alpha1-044036.pkl) |
# |----------|---------|---------------------------------------------------------------------------------------------------------|
# | EDM2-L   | SiD     | [edm2_img512_l_sid_alpha1-203788.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_l_sid_alpha1-203788.pkl) |
# | EDM2-L   | SiDA    | [edm2_img512_l_sida_alpha1-166918.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_l_sida_alpha1-166918.pkl) |
# | EDM2-L   | SiD²A   | [edm2_img512_l_sid2a_alpha1-050182.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_l_sid2a_alpha1-050182.pkl) |
# |----------|---------|---------------------------------------------------------------------------------------------------------|
# | EDM2-XL  | SiD     | [edm2_img512_xl_sid_alpha1-234004.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid_alpha1-234004.pkl) |
# | EDM2-XL  | SiDA    | [edm2_img512_xl_sid2a_alpha1-024578.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-024578.pkl) |
# | EDM2-XL  | SiD²A   | [edm2_img512_xl_sid2a_alpha1-079495.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-079495.pkl) |
# |----------|---------|---------------------------------------------------------------------------------------------------------|
# | EDM2-XXL | SiD     | [edm2_img512_xxl_sida_alpha1-077932.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xxl_sida_alpha1-077932.pkl) |
# | EDM2-XXL | SiDA    | [edm2_img512_xxl_sida_alpha1-089816.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xxl_sida_alpha1-089816.pkl) |
# | EDM2-XXL | SiD²A   | [edm2_img512_xxl_sid2a_alpha1-029812.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xxl_sid2a_alpha1-029812.pkl) |



########################## edm2-xs #####################
if [ "$model" = 'imagenet512-xs-sid' ]; then
    python -m torch.distributed.run --nproc_per_node=4 sida_edm2_train.py \
    --train_mode 0 \
    --use_gan 0 \
    --cond 1 \
    --alpha 1 \
    --fp16 1 \
    --init_sigma 2.5 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl' \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --data '/data/datasets/img512_labels/dataset.json' \
    --metrics fid50k_full \
    --arch 'edm2-img512-xs' \
    --outdir '/data/image_experiment/edm2-xs-metrics-sid' \
    --nosubdir 1 \
    --sid_model 'https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xs_sid_alpha1-249360.pkl'

elif [ "$model" = 'imagenet512-xs-sida' ]; then
    python -m torch.distributed.run --nproc_per_node=4 sida_edm2_train.py \
    --train_mode 0 \
    --use_gan 0 \
    --cond 1 \
    --alpha 1 \
    --fp16 1 \
    --init_sigma 2.5 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl' \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --data '/data/datasets/img512_labels/dataset.json' \
    --metrics fid50k_full \
    --arch 'edm2-img512-xs' \
    --outdir '/data/image_experiment/edm2-xs-metrics-sida' \
    --nosubdir 1 \
    --sid_model 'https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xs_sida_alpha1-166410.pkl'

elif [ "$model" = 'imagenet512-xs-sid-sida' ]; then
    python -m torch.distributed.run --nproc_per_node=4 sida_edm2_train.py \
    --train_mode 0 \
    --use_gan 0 \
    --cond 1 \
    --alpha 1 \
    --fp16 1 \
    --init_sigma 2.5 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl' \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --data '/data/datasets/img512_labels/dataset.json' \
    --metrics fid50k_full \
    --arch 'edm2-img512-xs' \
    --outdir '/data/image_experiment/edm2-xs-metrics-sid-sida' \
    --nosubdir 1 \
    --sid_model 'https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xs_sid2a_alpha1-028674.pkl'

########################## edm2-s #####################
elif [ "$model" = 'imagenet512-s-sid' ]; then
    python -m torch.distributed.run --nproc_per_node=4 sida_edm2_train.py \
    --train_mode 0 \
    --use_gan 0 \
    --cond 1 \
    --alpha 1 \
    --fp16 1 \
    --init_sigma 2.5 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl' \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --data '/data/datasets/img512_labels/dataset.json' \
    --metrics fid50k_full \
    --arch 'edm2-img512-s' \
    --outdir '/data/image_experiment/edm2-s-metrics-sid' \
    --nosubdir 1 \
    --sid_model 'https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_s_sid_alpha1-192010.pkl'

elif [ "$model" = 'imagenet512-s-sida' ]; then
    python -m torch.distributed.run --nproc_per_node=4 sida_edm2_train.py \
    --train_mode 0 \
    --use_gan 0 \
    --cond 1 \
    --alpha 1 \
    --fp16 1 \
    --init_sigma 2.5 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl' \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --data '/data/datasets/img512_labels/dataset.json' \
    --metrics fid50k_full \
    --arch 'edm2-img512-s' \
    --outdir '/data/image_experiment/edm2-s-metrics-sida' \
    --nosubdir 1 \
    --sid_model 'https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_s_sida_alpha1-117764.pkl'

elif [ "$model" = 'imagenet512-s-sid-sida' ]; then
    python -m torch.distributed.run --nproc_per_node=4 sida_edm2_train.py \
    --train_mode 0 \
    --use_gan 0 \
    --cond 1 \
    --alpha 1 \
    --fp16 1 \
    --init_sigma 2.5 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl' \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --data '/data/datasets/img512_labels/dataset.json' \
    --metrics fid50k_full \
    --arch 'edm2-img512-s' \
    --outdir '/data/image_experiment/edm2-s-metrics-sid-sida' \
    --nosubdir 1 \
    --sid_model 'https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_s_sid2a_alpha1-075268.pkl'

########################## edm2-m #####################
elif [ "$model" = 'imagenet512-m-sid' ]; then
    python -m torch.distributed.run --nproc_per_node=4 sida_edm2_train.py \
    --train_mode 0 \
    --use_gan 0 \
    --cond 1 \
    --alpha 1 \
    --fp16 1 \
    --init_sigma 2.5 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl' \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --data '/data/datasets/img512_labels/dataset.json' \
    --metrics fid50k_full \
    --arch 'edm2-img512-m' \
    --outdir '/data/image_experiment/edm2-m-metrics-sid' \
    --nosubdir 1 \
    --sid_model 'https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_m_sid_alpha1-226830.pkl'

elif [ "$model" = 'imagenet512-m-sida' ]; then
    python -m torch.distributed.run --nproc_per_node=4 sida_edm2_train.py \
    --train_mode 0 \
    --use_gan 0 \
    --cond 1 \
    --alpha 1 \
    --fp16 1 \
    --init_sigma 2.5 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl' \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --data '/data/datasets/img512_labels/dataset.json' \
    --metrics fid50k_full \
    --arch 'edm2-img512-m' \
    --outdir '/data/image_experiment/edm2-m-metrics-sida' \
    --nosubdir 1 \
    --sid_model 'https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_m_sida_alpha1-115716.pkl'

elif [ "$model" = 'imagenet512-m-sid-sida' ]; then
    python -m torch.distributed.run --nproc_per_node=4 sida_edm2_train.py \
    --train_mode 0 \
    --use_gan 0 \
    --cond 1 \
    --alpha 1 \
    --fp16 1 \
    --init_sigma 2.5 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl' \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --data '/data/datasets/img512_labels/dataset.json' \
    --metrics fid50k_full \
    --arch 'edm2-img512-m' \
    --outdir '/data/image_experiment/edm2-m-metrics-sid-sida' \
    --nosubdir 1 \
    --sid_model 'https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_m_sid2a_alpha1-044036.pkl'

########################## edm2-l #####################
elif [ "$model" = 'imagenet512-l-sid' ]; then
    python -m torch.distributed.run --nproc_per_node=4 sida_edm2_train.py \
    --train_mode 0 \
    --use_gan 0 \
    --cond 1 \
    --alpha 1 \
    --fp16 1 \
    --init_sigma 2.5 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl' \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --data '/data/datasets/img512_labels/dataset.json' \
    --metrics fid50k_full \
    --arch 'edm2-img512-l' \
    --outdir '/data/image_experiment/edm2-l-metrics-sid' \
    --nosubdir 1 \
    --sid_model 'https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_l_sid_alpha1-203788.pkl'

elif [ "$model" = 'imagenet512-l-sida' ]; then
    python -m torch.distributed.run --nproc_per_node=4 sida_edm2_train.py \
    --train_mode 0 \
    --use_gan 0 \
    --cond 1 \
    --alpha 1 \
    --fp16 1 \
    --init_sigma 2.5 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl' \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --data '/data/datasets/img512_labels/dataset.json' \
    --metrics fid50k_full \
    --arch 'edm2-img512-l' \
    --outdir '/data/image_experiment/edm2-l-metrics-sida' \
    --nosubdir 1 \
    --sid_model 'https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_l_sida_alpha1-166918.pkl'

elif [ "$model" = 'imagenet512-l-sid-sida' ]; then
    python -m torch.distributed.run --nproc_per_node=4 sida_edm2_train.py \
    --train_mode 0 \
    --use_gan 0 \
    --cond 1 \
    --alpha 1 \
    --fp16 1 \
    --init_sigma 2.5 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl' \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --data '/data/datasets/img512_labels/dataset.json' \
    --metrics fid50k_full \
    --arch 'edm2-img512-l' \
    --outdir '/data/image_experiment/edm2-l-metrics-sid-sida' \
    --nosubdir 1 \
    --sid_model 'https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_l_sid2a_alpha1-050182.pkl'

########################## edm2-xl #####################
elif [ "$model" = 'imagenet512-xl-sid' ]; then
    python -m torch.distributed.run --nproc_per_node=4 sida_edm2_train.py \
    --train_mode 0 \
    --use_gan 0 \
    --cond 1 \
    --alpha 1 \
    --fp16 1 \
    --init_sigma 2.5 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl' \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --data '/data/datasets/img512_labels/dataset.json' \
    --metrics fid50k_full \
    --arch 'edm2-img512-xl' \
    --outdir '/data/image_experiment/edm2-xl-metrics-sid' \
    --nosubdir 1 \
    --sid_model 'https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid_alpha1-234004.pkl'

elif [ "$model" = 'imagenet512-xl-sida' ]; then
    python -m torch.distributed.run --nproc_per_node=4 sida_edm2_train.py \
    --train_mode 0 \
    --use_gan 0 \
    --cond 1 \
    --alpha 1 \
    --fp16 1 \
    --init_sigma 2.5 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl' \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --data '/data/datasets/img512_labels/dataset.json' \
    --metrics fid50k_full \
    --arch 'edm2-img512-xl' \
    --outdir '/data/image_experiment/edm2-xl-metrics-sida' \
    --nosubdir 1 \
    --sid_model 'https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-024578.pkl'

elif [ "$model" = 'imagenet512-xl-sid-sida' ]; then
    python -m torch.distributed.run --nproc_per_node=4 sida_edm2_train.py \
    --train_mode 0 \
    --use_gan 0 \
    --cond 1 \
    --alpha 1 \
    --fp16 1 \
    --init_sigma 2.5 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl' \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --data '/data/datasets/img512_labels/dataset.json' \
    --metrics fid50k_full \
    --arch 'edm2-img512-xl' \
    --outdir '/data/image_experiment/edm2-xl-metrics-sid-sida' \
    --nosubdir 1 \
    --dropout 0.1 \
    --augment 0 \
    --sid_model 'https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-079495.pkl'  

########################## edm2-xxl #####################
elif [ "$model" = 'imagenet512-xxl-sid' ]; then
    python -m torch.distributed.run --nproc_per_node=4 sida_edm2_train.py \
    --train_mode 0 \
    --use_gan 0 \
    --cond 1 \
    --alpha 1 \
    --fp16 1 \
    --init_sigma 2.5 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl' \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --data '/data/datasets/img512_labels/dataset.json' \
    --metrics fid50k_full \
    --arch 'edm2-img512-xxl' \
    --outdir '/data/image_experiment/edm2-xxl-metrics-sid' \
    --nosubdir 1 \
    --sid_model 'https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xxl_sida_alpha1-077932.pkl'

elif [ "$model" = 'imagenet512-xxl-sida' ]; then
    python -m torch.distributed.run --nproc_per_node=4 sida_edm2_train.py \
    --train_mode 0 \
    --use_gan 0 \
    --cond 1 \
    --alpha 1 \
    --fp16 1 \
    --init_sigma 2.5 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl' \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --data '/data/datasets/img512_labels/dataset.json' \
    --metrics fid50k_full \
    --arch 'edm2-img512-xxl' \
    --outdir '/data/image_experiment/edm2-xxl-metrics-sida' \
    --nosubdir 1 \
    --sid_model 'https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xxl_sida_alpha1-089816.pkl'

elif [ "$model" = 'imagenet512-xxl-sid-sida' ]; then
    python -m torch.distributed.run --nproc_per_node=1 sida_edm2_train.py \
    --train_mode 0 \
    --use_gan 0 \
    --cond 1 \
    --alpha 1 \
    --fp16 1 \
    --init_sigma 2.5 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl' \
    --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
    --data '/data/datasets/img512_labels/dataset.json' \
    --metrics fid50k_full \
    --arch 'edm2-img512-xxl' \
    --outdir '/data/image_experiment/edm2-xxl-metrics-sid-sida' \
    --nosubdir 1 \
    --sid_model 'https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xxl_sid2a_alpha1-029812.pkl'  

else
    echo "Invalid dataset specified"
    exit 1
fi

