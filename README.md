## Distilling Pretrained Diffusion-Based Generative Models

This repository contains the code necessary to replicate the results of the "Score Identity Distillation: Exponentially Fast Distillation of Pretrained Diffusion Models for One-Step Generation" paper. The technique, Score Identity Distillation (SiD), is applied to distill pretrained EDM diffusion models. SiD demonstrates superior performance compared to the teacher EDM model across all datasets, with the exception of ImageNet 64x64.

### Code Contributions
- **Mingyuan Zhou**: Led the project and wrote the majority of the code.
- **Huangjie Zheng, Zhendong Wang, Hai Huang**: Worked closely with Mingyuan Zhou, co-developing essential components and writing various subfunctions.


The one-step generators produced by SiD are provided in ...

Ready to Use:
- sid_train.py
- generator_onestep.py

To DO:
- Test sid_metrics.py
- Test sid_generator.py

