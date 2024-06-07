# Distilling Pretrained Diffusion-Based Generative Models

This repository contains the code necessary to replicate the results of the "Score identity Distillation: Exponentially Fast Distillation of Pretrained Diffusion Models for One-Step Generation" paper. The technique, Score identity Distillation (SiD), is applied to distill pretrained EDM diffusion models. SiD demonstrates superior performance compared to the teacher EDM model across all datasets, with the exception of ImageNet 64x64. It can achieve the following Fréchet Inception Distances (FID):

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

After activating the environment, you can run the scripts or use the modules provided in the repository. Example:

```bash
sh run_sid.sh
```

### Training
- sid_train.py

### Generation

- generator_onestep.py
- sid_metrics.py
- sid_generator.py


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

This project uses the following license: [CC4.0](README.md).
