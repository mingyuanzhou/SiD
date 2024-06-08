# Distilling Pretrained Diffusion-Based Generative Models

This repository contains the code necessary to replicate the findings of our ICML 2024 paper titled "Score identity Distillation: Exponentially Fast Distillation of Pretrained Diffusion Models for One-Step Generation." The technique, Score identity Distillation (SiD), is used to distill pretrained EDM diffusion models.

SiD operates as a data-free distillation method but still demonstrates superior performance compared to the teacher EDM model across most datasets, with the notable exception of ImageNet 64x64. It outperforms all previous diffusion distillation approaches—whether one-step or few-step, data-free or training data-dependent—in terms of generation quality. This achievement sets new standards for efficiency and effectiveness in diffusion distillation.

It achieves the following Fréchet Inception Distances (FID):

| Dataset              | FID   |
|----------------------|-------|
| CIFAR10 Unconditional| 1.923 |
| CIFAR10 Conditional  | 1.710 |
| ImageNet 64x64       | 1.524 |
| FFHQ 64x64           | 1.550 |
| AFHQ-v2 64x64        | 1.628 |


## Prerequisites

Before you begin, ensure you have met the following requirements:
* You have installed the latest version of [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
* You have a `Windows/Linux/Mac` machine.

## Installation

To install the necessary packages and set up the environment, follow these steps:

### Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/mingyuanzhou/SiD.git
cd SiD
```

### Create the Conda Environment

To create the Conda environment with all the required dependencies, run:

```bash
conda env create -f environment.yaml
```

This command will read the `environment.yaml` file in the repository, which contains all the necessary package information.

### Activate the Environment

After creating the environment, you can activate it by running:

```bash
conda activate sid
```

### Prepare the Datasets

Follow the instructions detailed in the [EDM codebase](https://github.com/NVlabs/edm/tree/main?tab=readme-ov-file#preparing-datasets) to prepare the training datasets. Once prepared, place them into the `/data/datasets/` folder:

- `cifar10-32x32.zip`
- `imagenet-64x64.zip`
- `ffhq-64x64.zip`
- `afhqv2-64x64.zip`

**Note:** Although a training dataset is not necessary for distilling the pretrained EDM model, it is used in our code to calculate evaluation metrics such as FID and Inception Score. Optionally, you can create a dummy dataset and either disable the evaluation code if you wish to run the SID distillation code without these metrics, or provide an npz file of the training dataset if you need to compute these metrics.

## Usage


### Training
After activating the environment, you can run the scripts or use the modules provided in the repository. Example:

```bash
sh run_sid.sh 'cifar10-uncond'
```

Adjust the --batch-gpu parameter according to your GPU memory limitations. The default setting for cifar10-uncond consumes less than 10 GB of memory per GPU.

### Generation

Generate example images:

#### Generate images and save them as out/*.png and out.npz

- Using a single GPU
```bash
    python generate_onestep.py --outdir=image_experiment/sid-train-runs/out --seeds=0-63 --batch=64 \
        --network=<network_path>
```
- Using multiple GPU
```bash
    torchrun --standalone --nproc_per_node=2 generate_onestep.py --outdir=image_experiment/sid-train-runs/out --seeds=0-999 --batch=64 \\
        --network=<network_path>
```
#### Generate and save 50,000 images, and compute FID using the saved images
- sid_generator.py

- Use a single GPU
```python sid_generate.py --outdir=image_experiment/out --seeds=0-49999 --batch=128 --network='https://huggingface.co/UT-Austin-PML/SiD/resolve/main/cifar10-uncond/alpha1.2/network-snapshot-1.200000-403968.pkl' --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz ```

- Use four GPUs
```torchrun --standalone --nproc_per_node=4 sid_generate.py --outdir=out --seeds=0-49999 --batch=128 --network='https://huggingface.co/UT-Austin-PML/SiD/resolve/main/imagenet64/alpha1.2/network-snapshot-1.200000-939176.pkl' --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz```

#### Perform 10 random trials, each trial compute FID and IS using 50,000 randomly generated images. 
- sid_metrics.py

```torchrun --standalone --nproc_per_node=7  sid_metrics.py  --cond=False --metrics='fid50k_full,is50k' --network='https://huggingface.co/UT-Austin-PML/SiD/resolve/main/cifar10-uncond/alpha1.2/network-snapshot-1.200000-403968.pkl' --data='/data/datasets/cifar10-32x32.zip' --data_stat='https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz'```

To Do:
allow the --data option to be removed, only using --data_stat is enough

## Checkpoints of one-step generators produced by SiD

The one-step generators produced by SiD are provided in [huggingface/UT-Austin-PML/SiD](https://huggingface.co/UT-Austin-PML/SiD/tree/main)


### Acknowledgements

We extend our gratitude to the authors of the **EDM paper** for sharing their code, which served as the foundational framework for developing SiD. The repository can be found here: [NVlabs/edm](https://github.com/NVlabs/edm).

Additionally, we are thankful to the authors of the **Diff Instruct paper** for making their code available. Their contributions have been instrumental in integrating the evaluation pipeline into our training iterations. Their repository is accessible here: [pkulwj1994/diff_instruct](https://github.com/pkulwj1994/diff_instruct).



### Code Contributions
- **Mingyuan Zhou**: Led the project and wrote the majority of the code.
- **Huangjie Zheng, Zhendong Wang, Hai Huang**: Worked closely with Mingyuan Zhou, co-developing essential components and writing various subfunctions.


## Contributing to the Project

To contribute to this project, follow these steps:

1. Fork this repository.
2. Create a new branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively, see the GitHub documentation on [creating a pull request](https://help.github.com/articles/creating-a-pull-request/).

## Contact

If you want to contact me you can reach me at `mingyuan.zhou@mccombs.utexas.edu`.

## License

This project uses the following license: [Apache-2.0 license](README.md).