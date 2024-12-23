# Distilling Pretrained Diffusion-Based Generative Models with SiDA (SiD, SiDA, or SiD²A)

This repository contains the code to reproduce the results from our paper, **"Adversarial Score identity Distillation: Rapidly Surpassing the Teacher in One Step,"** available at [arXiv:2410.14919](https://arxiv.org/abs/2410.14919). The introduced methods—Score identity Distillation (SiD), Adversarial SiD (SiDA), and SiD-Initialized SiDA (SiD²A)—facilitate the distillation of pretrained EDM and EDM2 diffusion models.


## Citations

If you find this repository helpful or use our work in your research, please consider citing our paper:

```bibtex
@article{zhou2024adversarial,
  title={Adversarial Score Identity Distillation: Rapidly Surpassing the Teacher in One Step},
  author={Zhou, Mingyuan and Zheng, Huangjie and Gu, Yi and Wang, Zhendong and Huang, Hai},
  journal={arXiv preprint arXiv:2410.14919},
  year={2024}
}
```

Our work on SiDA builds on prior research, including SiD and SiD-LSA. If relevant, you may also want to cite the following:

### SiD: Score identity Distillation

```bibtex
@inproceedings{zhou2024score,
  title={Score Identity Distillation: Exponentially Fast Distillation of Pretrained Diffusion Models for One-Step Generation},
  author={Mingyuan Zhou and Huangjie Zheng and Zhendong Wang and Mingzhang Yin and Hai Huang},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

### SiD-LSG: SiD with Long and Short Guidance for One-step Text-to-Image Generation

```bibtex
@article{zhou2024long,
  title={Long and Short Guidance in Score Identity Distillation for One-Step Text-to-Image Generation},
  author={Mingyuan Zhou and Zhendong Wang and Huangjie Zheng and Hai Huang},
  journal={arXiv preprint arXiv:2406.01561},
  url={https://arxiv.org/abs/2406.01561},
  year={2024}
}
```


## State-of-the-Art Performance

SiDA achieves state-of-the-art performance, surpassing the teacher EDM/EDM2 models across all datasets and model sizes. Remarkably, this is accomplished in a single step, without relying on classifier-free guidance. SiDA sets new benchmarks for efficiency and effectiveness in diffusion distillation.

### Fréchet Inception Distance (FID) Results
SiDA achieves the following FID scores when distilling EDM:

| Dataset               | FID   |
|-----------------------|-------|
| CIFAR10 (Unconditional)| 1.499 |
| CIFAR10 (Conditional)  | 1.396 |
| ImageNet 64x64         | 1.110 |
| FFHQ 64x64             | 1.040 |
| AFHQ-v2 64x64          | 1.276 |

For EDM2 models pretrained on ImageNet 512x512, SiDA achieves the following FID scores:

| Teacher Model (Parameter Count) | FID   |
|---------------------------------|-------|
| EDM2-XS (125M)                  | 2.156 |
| EDM2-S  (280M)                  | 1.669 |
| EDM2-M  (498M)                  | 1.488 |
| EDM2-L  (777M)                  | 1.413 |
| EDM2-XL (1.1B)                  | 1.379 |
| EDM2-XXL (1.5B)                 | 1.366 |




## Prerequisites

Before you begin, ensure you have met the following requirements:
* You have installed the latest version of [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
* You have a `Windows/Linux/Mac` machine.

## Installation

To install the necessary packages and set up the environment, follow these steps:

### Clone the Repository

First, clone the repository to your local machine:

```bash
git clone --branch sida https://github.com/mingyuanzhou/SiD.git
cd SiD
git branch
```

The output should show * sida, indicating that you're on the correct branch.

### Create the Conda Environment

To create the Conda environment with all the required dependencies, run:

```bash
conda env create -f environment.yaml
```

This command will read the `environment.yaml` file in the repository, which contains all the necessary package information.

### Activate the Environment

After creating the environment, you can activate it by running:

```bash
conda activate sida
```
### Prepare the Datasets

To prepare the training datasets needed to distill EDM, follow the instructions provided in the [EDM codebase](https://github.com/NVlabs/edm/tree/main?tab=readme-ov-file#preparing-datasets). Once the datasets are ready, place them in the `/data/datasets/` folder with the following names:

- `cifar10-32x32.zip`
- `imagenet-64x64.zip`
- `ffhq-64x64.zip`
- `afhqv2-64x64.zip`

To prepare the ImageNet 512x512 training dataset needed to distill EDM2, follow the instructions provided in the [EDM2 codebase](https://github.com/NVlabs/edm2/tree/main?tab=readme-ov-file#preparing-datasets). Once the dataset is ready, place them in the `/data/datasets/` folder with the following names:

- `img512-sd.zip`


**Important Notes:**

- **For SiD:**
  - A training dataset is not required for distilling pretrained EDM models. However, it is used in the code to compute evaluation metrics like FID and Inception Score.
  - If you do not need these metrics, you can either:
    - Provide a dummy dataset.
    - Disable the evaluation code to run SiD distillation without metrics.
  - Alternatively, if you want to compute these metrics, ensure you provide an `.npz` file of the training dataset.

- **For SiDA or SiD-SiDA:**
  - A training dataset is mandatory as it is actively used in the distillation process.


## Usage


### Distilling EDM

Once the environment is activated, you can start training by running the provided scripts or modules. Here are examples for each method:

- **SiD:**
  ```bash
  sh run_sid_v1.sh 'cifar10-uncond'
  ```

- **SiDA:**
  ```bash
  sh run_sida.sh 'cifar10-uncond'
  ```

- **SiD-SiDA (SiD²A):**
  ```bash
  sh run_sid_sida.sh 'cifar10-uncond'
  ```
  

### Distilling EDM2 (to be added)

Once the environment is activated, you can start training by running the provided scripts or modules. Here are examples for each method:

- **SiD:**
  ```bash
  sh run_sid_v1.sh 'imagenet-edm2-xs'
  ```

- **SiDA:**
  ```bash
  sh run_sida.sh 'imagenet-edm2-xs'
  ```

- **SiD-SiDA (SiD²A):**
  ```bash
  sh run_sid_sida.sh 'imagenet-edm2-xs'
  ```

### Key Revisions:
1. **Improved Readability:** Organized the examples into a list format for clarity.
2. **Consistent Naming:** Used consistent casing and styling for terms like "SiD" and "SiDA."
3. **Enhanced Presentation:** Highlighted the training commands with proper markdown styling.

Adjust the --batch-gpu parameter according to your GPU memory limitations. The default setting for cifar10-uncond consumes less than 10 GB of memory per GPU.

### Generation

Generate example images:

#### Generate images and save them as out/*.png and out.npz

- Using a single GPU
```bash
python generate_onestep.py --outdir=image_experiment/sid-train-runs/out --seeds=0-63 --batch=64 --network=<network_path>
```
- Using multiple GPU
```bash
torchrun --standalone --nproc_per_node=2 generate_onestep.py --outdir=image_experiment/sid-train-runs/out --seeds=0-999 --batch=64 --network=<network_path>
```

### Evaluations

For ImageNet, there are two different versions of the training data, each associated with its own set of reference statistics. To ensure apples-to-apples comparisons between EDM and its distilled generators with other diffusion models, `imagenet-64x64.npz` should be used for computing FID (Fréchet Inception Distance). Conversely, for computing Precision and Recall, `VIRTUAL_imagenet64_labeled.npz` should be utilized.

`imagenet-64x64.npz` is available at [NVIDIA](https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz).

`VIRTUAL_imagenet64_labeled.npz` is available at [OpenAI](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/64/VIRTUAL_imagenet64_labeled.npz).

#### Use `sid_generator.py` to generate and save 50,000 images, and compute FID using the saved images

##### Use a single GPU
```bash 
python sid_generate.py --outdir=image_experiment/out --seeds=0-49999 --batch=128 --network='https://huggingface.co/UT-Austin-PML/SiD/resolve/main/cifar10-uncond/alpha1.2/network-snapshot-1.200000-403968.pkl' --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz 
```

##### Use four GPUs
```bash 
torchrun --standalone --nproc_per_node=4 sid_generate.py --outdir=out --seeds=0-49999 --batch=128 --network='https://huggingface.co/UT-Austin-PML/SiD/resolve/main/imagenet64/alpha1.2/network-snapshot-1.200000-939176.pkl' --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz
```


#### Use `sid_metrics.py` to perform 10 random trials, each trial computes the metrics using 50,000 randomly generated images

##### Compute FID and/or IS 

```bash
torchrun --standalone --nproc_per_node=4  sid_metrics.py  --cond=False --metrics='fid50k_full,is50k' --network='https://huggingface.co/UT-Austin-PML/SiD/resolve/main/cifar10-uncond/alpha1.2/network-snapshot-1.200000-403968.pkl' --data='/data/datasets/cifar10-32x32.zip'
```


```bash
torchrun --standalone --nproc_per_node=4  sid_metrics.py  --cond=True --metrics='fid50k_full' --network='https://huggingface.co/UT-Austin-PML/SiD/resolve/main/imagenet64/alpha1.2/network-snapshot-1.200000-939176.pkl' --data='/data/datasets/imagenet-64x64.zip' --data_stat='https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz' 
```


##### Compute Precision and Recall for ImageNet

```bash
torchrun --standalone --nproc_per_node=4  sid_metrics.py  --cond=True --metrics='pr50k3_full' --network='https://huggingface.co/UT-Austin-PML/SiD/resolve/main/imagenet64/alpha1.2/network-snapshot-1.200000-939176.pkl' --data='/data/datasets/imagenet-64x64.zip' --data_stat='https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/64/VIRTUAL_imagenet64_labeled.npz'
```


## Checkpoints of one-step generators produced by SiD

The one-step generators produced by SiD are provided in [huggingface/UT-Austin-PML/SiD](https://huggingface.co/UT-Austin-PML/SiD/tree/main)


### Acknowledgements

We extend our gratitude to the authors of the **EDM paper** and **EDM2 paper**for sharing their code, which served as the foundational frameworks for developing SiDA. The EDM repository can be found here: [NVlabs/edm](https://github.com/NVlabs/edm). The EDM2 repository can be found here: [NVlabs/edm2](https://github.com/NVlabs/edm2).


### Code Contributions
- **Mingyuan Zhou**: Led the project and wrote the majority of the code.
- **Huangjie Zheng, Yi Gu, Zhendong Wang, Hai Huang**: Worked with Mingyuan Zhou to develop subfunctions and verify the code and results.


## Contributing to the Project

To contribute to this project, follow these steps:

1. Fork this repository.
2. Create a new branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively, see the GitHub documentation on [creating a pull request](https://help.github.com/articles/creating-a-pull-request/).

## Contact

If you want to contact me you can reach me at `mingyuanzhou@gmail.com`.

## License

This project uses the following license: [Apache-2.0 license](README.md).