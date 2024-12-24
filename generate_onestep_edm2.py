# Copyright (c) 2024, Mingyuan Zhou. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist


from sida_training_edm2.vae_edm2_utils import  load_sd_vae
#----------------------------------------------------------------------------
# One-step generator that allows specifying a different random seed for each generated sample

raw_mean = np.array([5.81, 3.25, 0.12, -2.15], dtype=np.float32) # Assumed mean of the raw latents.
raw_std = np.array([4.17, 4.62, 3.71, 3.28], dtype=np.float32) # Assumed standard deviation of the raw latents.
final_mean = np.float32(0) # Desired mean of the final latents.
final_std = np.float32(0.5) # Desired standard deviation of the final latents.
scale = final_std / raw_std
bias = final_mean - raw_mean * scale


def run_generator(G,z, c, init_sigma, opts):
    with torch.no_grad():
        # Ensure all necessary attributes are in opts and are of tensor type
        scale = torch.tensor(opts.scale, dtype=torch.float32, device=z.device) if hasattr(opts, 'scale') else torch.tensor(1.0, device=z.device)
        bias = torch.tensor(opts.bias, dtype=torch.float32, device=z.device) if hasattr(opts, 'bias') else torch.tensor(0.0, device=z.device)
        vae = opts.vae if hasattr(opts, 'vae') else None

        # Initialize sigma tensor and pass through G
        init_sigma_tensor = init_sigma * torch.ones(z.shape[0], 1, 1, 1, device=z.device)
        img = G(z, init_sigma_tensor, c, augment_labels=torch.zeros(z.shape[0], 9, device=z.device))

        if vae is not None:
            vae.to(z.device)  # Ensure VAE is on the same device as the input tensor
            #img = img.to(torch.float32)
            img = (img - bias.reshape(1, -1, 1, 1)) / scale.reshape(1, -1, 1, 1)
            img = vae.decode(img).sample  # Ensure `.sample` is compatible with `vae.decode`

        # Final scaling to uint8 for image representation
        img = img.clamp(0, 1).mul(255).to(torch.uint8)
        #print(img.shape)  # For debugging purposes

    return img

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def compress_to_npz(folder_path, num=50000):
    # Get the list of all files in the folder
    npz_path = f"{folder_path}.npz"
    file_names = os.listdir(folder_path)

    # Filter the list of files to include only images
    file_names = [file_name for file_name in file_names if file_name.endswith(('.png', '.jpg', '.jpeg'))]
    num = min(num, len(file_names))
    file_names = file_names[:num]

    # Initialize a dictionary to hold image arrays and their filenames
    samples = []

    # Iterate through the files, load each image, and add it to the dictionary with a progress bar
    for file_name in tqdm.tqdm(file_names, desc=f"Compressing images to {npz_path}"):
        # Create the full path to the image file
        file_path = os.path.join(folder_path, file_name)
        
        # Read the image using PIL and convert it to a NumPy array
        image = PIL.Image.open(file_path)
        image_array = np.asarray(image).astype(np.uint8)
        
        samples.append(image_array)
    samples = np.stack(samples)

    # Save the images as a .npz file
    np.savez(npz_path, arr_0=samples)
    print(f"Images from folder {folder_path} have been saved as {npz_path}")

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--num', 'num_fid_samples',  help='Maximum num of images', metavar='INT',                             type=click.IntRange(min=1), default=50000, show_default=True)
@click.option('--sigma_G', 'sigma_G',      help='Stoch. noise std', metavar='FLOAT',                                type=float, default=2.5, show_default=True)

def main(network_pkl, outdir, subdirs, seeds, class_idx, max_batch_size, num_fid_samples, sigma_G, device=torch.device('cuda')):
    """Generate random images using SiD".

    Examples:
    
    python generate_onestep_edm2.py --outdir=out --seeds=0-63  --batch=64 --network='/data/Austin-PML/SiD-EDM2/img512_edm2_xl_sida_alpha1-065032.pkl'
    torchrun --standalone --nproc_per_node=6 generate_onestep_edm2.py --outdir=out --seeds=0-49999 --batch=2 --outdir=/data/image_experiment/out  --batch=64 --network='/data/Austin-PML/SiD-EDM2/img512_edm2_xl_sida_alpha1-065032.pkl'
    \b
    # Generate 64 images and save them as out/*.png and out.npz
    python generate_onestep.py --outdir=out --seeds=0-63 --batch=64 \
        --network=<network_path>
        
    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate_onestep.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=<network_path>
        
        
    \b
    # Generate 64 images and save them as out/*.png and out.npz
    python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xs_class88 --seeds=0-63  --batch=2   --class=88 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xs_sid2a_alpha1-028674.pkl'
    
    python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xxl_class88 --seeds=0-63  --batch=2   --class=88 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xxl_sid2a_alpha1-029812.pkl'    
    
     python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xxl_class29 --seeds=0-63  --batch=2   --class=29 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xxl_sid2a_alpha1-029812.pkl'
    
    python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xl_class88 --seeds=0-63  --batch=2   --class=88 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-079495.pkl'  
    
    python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xl_class29 --seeds=0-63  --batch=2   --class=29 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-079495.pkl'  
    
    python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xl_class127 --seeds=0-63  --batch=2   --class=127 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-079495.pkl'  
    
    python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xl_class89 --seeds=0-63  --batch=2   --class=89 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-079495.pkl'  
       
    python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xl_class980 --seeds=0-63  --batch=2   --class=980 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-079495.pkl'  
       
    python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xl_class33 --seeds=0-63  --batch=2   --class=33 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-079495.pkl'  
    
    python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xl_class15 --seeds=0-63  --batch=2   --class=15 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-079495.pkl'  
      
    python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xl_class975 --seeds=0-63  --batch=2   --class=975 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-079495.pkl'  
    
    python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xl_class279 --seeds=0-63  --batch=2   --class=279 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-079495.pkl'  
    
    python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xl_class323 --seeds=0-63  --batch=2   --class=323 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-079495.pkl'  
    python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xl_class387 --seeds=0-63  --batch=2   --class=387 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-079495.pkl'  
    python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xl_class388 --seeds=0-63  --batch=2   --class=388 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-079495.pkl'  
    python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xl_class417 --seeds=0-63  --batch=2   --class=417 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-079495.pkl' 
    python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xl_class425 --seeds=0-63  --batch=2   --class=425 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-079495.pkl' 
    python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xl_class933 --seeds=0-63  --batch=2   --class=933 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-079495.pkl' 
    python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xl_class973 --seeds=0-63  --batch=2   --class=973 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-079495.pkl' 
    
    python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xl_class0 --seeds=0-63  --batch=2   --class=0 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-079495.pkl' 
    python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xl_class1 --seeds=0-63  --batch=2   --class=1 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-079495.pkl' 
    python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xl_class2 --seeds=0-63  --batch=2   --class=2 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-079495.pkl' 
    python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xl_class3 --seeds=0-63  --batch=2   --class=3 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-079495.pkl' 
    
    python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xl_class292 --seeds=0-63  --batch=2   --class=292 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xl_sid2a_alpha1-079495.pkl' 
    
    
    python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xxl_class0 --seeds=0-63  --batch=2   --class=0 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xxl_sid2a_alpha1-029812.pkl'
    python generate_onestep_edm2.py --outdir=/data/image_experiment/out_xxl_class1 --seeds=0-63  --batch=2   --class=1 --network='https://huggingface.co/UT-Austin-PML/SiDA/resolve/main/EDM2_distillation/edm2_img512_xxl_sid2a_alpha1-029812.pkl'
    
    """
    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]


    print(os.environ['CUDA_VISIBLE_DEVICES'])  # Should print: 2,3,4,5,6,7
    print(torch.cuda.device_count())  # Should print: 6
    
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)
        
    net.eval().requires_grad_(False).to(device)
    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    
    
    pretrained_vae_model_name_or_path = 'stabilityai/sd-vae-ft-mse'
    vae = load_sd_vae(pretrained_vae_model_name_or_path, device=device)
    opts = dnnlib.EasyDict()
    opts.vae = vae
    opts.scale = scale
    opts.bias = bias
    
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        #print(net.img_channels)
        #print(net.img_resolution)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        sigma = sigma_G*torch.ones([batch_size, 1, 1, 1], device=device)
        class_labels = None
        #print(net.label_dim)
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        #images = net(sigma_G*latents.to(torch.float32), sigma, class_labels)   #.to(torch.float32)

        images = run_generator(net,sigma_G*latents.to(torch.float32),class_labels,sigma_G,opts)
        #print(images.shape)
# def run_generator(z, c, init_sigma, opts):
#     with torch.no_grad():
#         # Ensure all necessary attributes are in opts and are of tensor type
#         scale = torch.tensor(opts.scale, dtype=torch.float32, device=z.device) if hasattr(opts, 'scale') else torch.tensor(1.0, device=z.device)
#         bias = torch.tensor(opts.bias, dtype=torch.float32, device=z.device) if hasattr(opts, 'bias') else torch.tensor(0.0, device=z.device)
#         vae = opts.vae if hasattr(opts, 'vae') else None

#         # Initialize sigma tensor and pass through G
#         init_sigma_tensor = init_sigma * torch.ones(z.shape[0], 1, 1, 1, device=z.device)
#         img = G(z, init_sigma_tensor, c, augment_labels=torch.zeros(z.shape[0], 9, device=z.device))

#         if vae is not None:
#             vae.to(z.device)  # Ensure VAE is on the same device as the input tensor
#             #img = img.to(torch.float32)
#             img = (img - bias.reshape(1, -1, 1, 1)) / scale.reshape(1, -1, 1, 1)
#             img = vae.decode(img).sample  # Ensure `.sample` is compatible with `vae.decode`

#         # Final scaling to uint8 for image representation
#         img = img.clamp(0, 1).mul(255).to(torch.uint8)
#         #print(img.shape)  # For debugging purposes

#     return img
        
        
        # Save images.
        #images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        images_np = images.permute(0, 2, 3, 1).cpu().numpy()
        #images_np = images.cpu().numpy()
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)

    # Done.
    torch.distributed.barrier()
    if dist.get_rank() == 0:
        compress_to_npz(outdir, num_fid_samples)
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
