
# Distilling Pretrained Diffusion-Based Generative Models with SiDA (SiD, SiDA, or SiD²A)

This repository contains the code to reproduce the results from our paper, **"Adversarial Score identity Distillation: Rapidly Surpassing the Teacher in One Step,"** available at [arXiv:2410.14919](https://arxiv.org/abs/2410.14919). The introduced methods—Score identity Distillation (SiD), Adversarial SiD (SiDA), and SiD-Initialized SiDA (SiD²A)—facilitate the distillation of pretrained EDM and EDM2 diffusion models.


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





### Prepare the Datasets

Follow the instructions provided in the [EDM codebase](https://github.com/NVlabs/edm/tree/main?tab=readme-ov-file#preparing-datasets) to prepare datasets and place them in `/data/datasets/`.

### Important Notes:

- For **SiD**, a training dataset is not required but is used to compute evaluation metrics like FID and IS.
- For **SiDA** and **SiD²A**, a training dataset is mandatory.


## Usage


### Distilling EDM

Once the environment is activated, you can start training by running the provided scripts or modules. Adjust the --batch-gpu parameter according to your GPU memory limitations. Here are examples for each method:

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

Here are examples for each method:

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


### Checkpoints of one-step generators produced by SiD and SiDA

Distilled one-step generators from our work are available at the following locations:

- **Distilled EDM Models with SiD:**  
  [https://huggingface.co/UT-Austin-PML/SiD/tree/main](https://huggingface.co/UT-Austin-PML/SiD/tree/main)

- **Distilled EDM Models with SiDA and SiD²A:**  
  [https://huggingface.co/UT-Austin-PML/SiDA/tree/main/EDM_distillation](https://huggingface.co/UT-Austin-PML/SiDA/tree/main/EDM_distillation)

- **Distilled EDM2 Models with SiD, SiDA, and SiD²A:**  
  [https://huggingface.co/UT-Austin-PML/SiDA/tree/main/EDM2_distillation](https://huggingface.co/UT-Austin-PML/SiDA/tree/main/EDM2_distillation)

#### Specific One-step Generators distilled from EDM
The following table provides links to specific checkpoints for EDM distillation:
| Dataset         | Method    | $\alpha$ | Checkpoint URL                                   |
|---|---|---|-----------------------------|
| CIFAR10-Uncond  | SiD       | 1.2      | [cifar10_uncond_sid_alpha1.2](https://huggingface.co/UT-Austin-PML/SiD/resolve/main/cifar10-uncond/alpha1.2/network-snapshot-1.200000-403968.pkl) |
| CIFAR10-Uncond  | SiDA      | 1.0      | [cifar10_uncond_sida_alpha1-217920.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM_distillation/cifar10_uncond_sida_alpha1-217920.pkl) |
| CIFAR10-Uncond  | SiD²A     | 1.0      | [cifar10_uncond_sid2a_alpha1-028160.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM_distillation/cifar10_uncond_sid2a_alpha1-028160.pkl) |
| CIFAR10-Uncond  | SiD²A     | 1.2      | [cifar10_uncond_sid2a_alpha1.2-022528.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM_distillation/cifar10_uncond_sid2a_alpha1.2-022528.pkl) |
|---|---|---|-----------------------------|
| CIFAR10-Cond    | SiD       | 1.2      | [cifar10_cond_sid_alpha1.2](https://huggingface.co/UT-Austin-PML/SiD/resolve/main/cifar10-cond/alpha1.2/network-snapshot-1.200000-713312.pkl) |
| CIFAR10-Cond    | SiDA      | 1.0      | [cifar10_cond_sida_alpha1-293184.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM_distillation/cifar10_cond_sida_alpha1-293184.pkl) |
| CIFAR10-Cond    | SiD²A     | 1.0      | [cifar10_cond_sid2a_alpha1-038912.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM_distillation/cifar10_cond_sid2a_alpha1-038912.pkl) |
| CIFAR10-Cond    | SiD²A     | 1.2      | [cifar10_cond_sid2a_alpha1.2-053760.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM_distillation/cifar10_cond_sid2a_alpha1.2-053760.pkl) |
|---|---|---|-----------------------------|
| ImageNet 64x64  | SiD       | 1.2      | [imagenet_sid_alpha1.2](https://huggingface.co/UT-Austin-PML/SiD/resolve/main/imagenet64/alpha1.2/network-snapshot-1.200000-939176.pkl) |
| ImageNet 64x64  | SiDA      | 1.0      | [imgnet64_sida_alpha1-249167.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM_distillation/imgnet64_sida_alpha1-249167.pkl) |
| ImageNet 64x64  | SiD²A     | 1.0      | [imgnet64_sid2a_alpha1-029499.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM_distillation/imgnet64_sid2a_alpha1-029499.pkl) |
| ImageNet 64x64  | SiD²A     | 1.2      | [imgnet64_sid2a_alpha1.2-021471.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM_distillation/imgnet64_sid2a_alpha1.2-021471.pkl) |
|---|---|---|-----------------------------|
| FFHQ64          | SiD       | 1.2      | [ffhq64_sid_alpha1.2](https://huggingface.co/UT-Austin-PML/SiD/resolve/main/ffhq64/alpha1.2/network-snapshot-1.200000-498176.pkl) |
| FFHQ64          | SiDA      | 1.0      | [ffhq64_sida_alpha1-185856.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM_distillation/ffhq64_sida_alpha1-185856.pkl) |
| FFHQ64          | SiD²A     | 1.0      | [ffhq64_sid2a_alpha1-168960.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM_distillation/ffhq64_sid2a_alpha1-168960.pkl) |
| FFHQ64          | SiD²A     | 1.2      | [ffhq64_sid2a_alpha1.2-080896.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM_distillation/ffhq64_sid2a_alpha1.2-080896.pkl) |
|---|---|---|-----------------------------|
| AFHQ64-v2       | SiD       | 1.2      | [afhq64_sid_alpha1.2](https://huggingface.co/UT-Austin-PML/SiD/resolve/main/afhq64/alpha1/network-snapshot-1.000000-371712.pkl) |
| AFHQ64-v2       | SiDA      | 1.0      | [afhq64-v2_sida_alpha1-127488.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM_distillation/afhq64-v2_sida_alpha1-127488.pkl) |
| AFHQ64-v2       | SiD²A     | 1.0      | [afhq64-v2_sid2a_alpha1-175104.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM_distillation/afhq64-v2_sid2a_alpha1-175104.pkl) |
| AFHQ64-v2       | SiD²A     | 1.2      | [afhq64-v2_sid2a_alpha1.2-175104.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM_distillation/afhq64-v2_sid2a_alpha1.2-175104.pkl) |


#### Specific One-step Generators for ImageNet 512x512 distilled from EDM2
The following table provides links to specific checkpoints for the distillation of EDM2 models pretrained on ImageNet 512x512:

| Model    | Method  | Checkpoint URL                                   |
|---|---|---|-----------------------------|
| EDM2-XS  | SiD     | [edm2_img512_xs_sid_alpha1-249360.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xs_sid_alpha1-249360.pkl) |
| EDM2-XS  | SiDA    | [edm2_img512_xs_sida_alpha1-166410.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xs_sida_alpha1-166410.pkl) |
| EDM2-XS  | SiD²A   | [edm2_img512_xs_sid2a_alpha1-028674.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xs_sid2a_alpha1-028674.pkl) |
|---|---|---|-----------------------------|
| EDM2-S   | SiD     | [edm2_img512_s_sid_alpha1-192010.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_s_sid_alpha1-192010.pkl) |
| EDM2-S   | SiDA    | [edm2_img512_s_sida_alpha1-117764.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_s_sida_alpha1-117764.pkl) |
| EDM2-S   | SiD²A   | [edm2_img512_s_sid2a_alpha1-075268.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_s_sid2a_alpha1-075268.pkl) |
|---|---|---|-----------------------------|
| EDM2-M   | SiD     | [edm2_img512_m_sid_alpha1-226830.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_m_sid_alpha1-226830.pkl) |
| EDM2-M   | SiDA    | [edm2_img512_m_sida_alpha1-115716.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_m_sida_alpha1-115716.pkl) |
| EDM2-M   | SiD²A   | [edm2_img512_m_sid2a_alpha1-044036.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_m_sid2a_alpha1-044036.pkl) |
|---|---|---|-----------------------------|
| EDM2-L   | SiD     | [edm2_img512_l_sid_alpha1-203788.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_l_sid_alpha1-203788.pkl) |
| EDM2-L   | SiDA    | [edm2_img512_l_sida_alpha1-166918.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_l_sida_alpha1-166918.pkl) |
| EDM2-L   | SiD²A   | [edm2_img512_l_sid2a_alpha1-050182.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_l_sid2a_alpha1-050182.pkl) |
|---|---|---|-----------------------------|
| EDM2-XL  | SiD     | [edm2_img512_xl_sid_alpha1-234004.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid_alpha1-234004.pkl) |
| EDM2-XL  | SiDA    | [edm2_img512_xl_sid2a_alpha1-024578.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-024578.pkl) |
| EDM2-XL  | SiD²A   | [edm2_img512_xl_sid2a_alpha1-079495.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-079495.pkl) |
|---|---|---|-----------------------------|
| EDM2-XXL | SiD     | [edm2_img512_xxl_sida_alpha1-077932.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xxl_sida_alpha1-077932.pkl) |
| EDM2-XXL | SiDA    | [edm2_img512_xxl_sida_alpha1-089816.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xxl_sida_alpha1-089816.pkl) |
| EDM2-XXL | SiD²A   | [edm2_img512_xxl_sid2a_alpha1-029812.pkl](https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xxl_sid2a_alpha1-029812.pkl) |


### Generation


#### Generate Images and Save Them as `out/*.png` and `out.npz`

You can generate images using the provided scripts, either for EDM or EDM2 models. Below are the instructions for both single and multiple GPU setups.

---

### Generating Images with EDM

1. **Using a Single GPU**  
   See comments inside `generate_onestep.py` for more details.

   ```bash
   python generate_onestep.py --outdir=image_experiment/sid-train-runs/out --seeds=0-63 --batch=64 --network=<network_path>
   ```

2. **Using Multiple GPUs**  
   Use the following command for distributed generation with multiple GPUs:

   ```bash
   torchrun --standalone --nproc_per_node=2 generate_onestep.py --outdir=image_experiment/sid-train-runs/out --seeds=0-999 --batch=64 --network=<network_path>
   ```

---

### Generating Images with EDM2

1. **Using a Single GPU**  
   See comments inside `generate_onestep_edm2.py` for more details. Examples for specific classes and networks are provided below.

   - **Class 88 (EDM2-XS):**
     ```bash
     python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xs_class88 --seeds=0-63 --batch=2 --class=88 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xs_sid2a_alpha1-028674.pkl'
     ```

   - **Class 88 (EDM2-XL):**
     ```bash
     python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xl_class88 --seeds=0-63 --batch=2 --class=88 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-079495.pkl'
     ```

   - **Class 88 (EDM2-XXL):**
     ```bash
     python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xxl_class88 --seeds=0-63 --batch=2 --class=88 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xxl_sid2a_alpha1-029812.pkl'
     ```

   - **Class 29 (EDM2-XXL):**
     ```bash
     python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xxl_class29 --seeds=0-63 --batch=2 --class=29 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xxl_sid2a_alpha1-029812.pkl'
     ```

---

### Notes:
- Adjust the `--seeds` and `--batch` values based on your desired output and available GPU memory.
- Replace `<network_path>` with the actual path or URL to the model checkpoint you want to use.
- For class-specific generations, specify the class index using the `--class` parameter.


### Evaluations

For ImageNet 64x64, there are two different versions of the training data, each associated with its own set of reference statistics. To ensure apples-to-apples comparisons between EDM and its distilled generators with other diffusion models, `imagenet-64x64.npz` should be used for computing FID (Fréchet Inception Distance). Conversely, for computing Precision and Recall, `VIRTUAL_imagenet64_labeled.npz` should be utilized.

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

## License

This project uses the following license: [Apache-2.0 license](README.md).