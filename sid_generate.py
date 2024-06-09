# Copyright (c) 2024, Mingyuan Zhou. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt


"""Generate random images using the techniques described in the paper
"Score identity Distillation: Exponentially Fast Distillation of
Pretrained Diffusion Models for One-Step Generation"."""

"""Compute FID by comparing the statistics of generated images with those of reference data."""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import socket
import re
import json
import click
import scipy.linalg
import torch
import dnnlib

from torch_utils import distributed as dist
from training import dataset


import click
import tqdm
import pickle
import numpy as np

import PIL.Image


#----------------------------------------------------------------------------

def sid_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,init_sigma=2.5
):
    z = latents.to(torch.float64) * init_sigma
    x = net(z, (init_sigma*torch.ones(z.shape[0],1,1,1)).to(z.device), class_labels).to(torch.float64)
    return x

def calculate_inception_stats(detector_url,
    image_path, num_expected=None, seed=0, max_batch_size=64,
    num_workers=3, prefetch_factor=2, device=torch.device('cuda'),
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load Inception-v3 model.
    # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    dist.print0('Loading Inception-v3 model...')
    #detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048
    with dnnlib.util.open_url(detector_url, verbose=(dist.get_rank() == 0)) as f:
        detector_net = pickle.load(f).to(device)

    # List images.
    dist.print0(f'Loading images from "{image_path}"...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)
    if num_expected is not None and len(dataset_obj) < num_expected:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_expected}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = ((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches, num_workers=num_workers, prefetch_factor=prefetch_factor)

    # Accumulate statistics.
    dist.print0(f'Calculating statistics for {len(dataset_obj)} images...')
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    for images, _labels in tqdm.tqdm(data_loader, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
        mu += features.sum(0)
        sigma += features.T @ features

    # Calculate grand totals.
    torch.distributed.all_reduce(mu)
    torch.distributed.all_reduce(sigma)
    mu /= len(dataset_obj)
    sigma -= mu.ger(mu) * len(dataset_obj)
    sigma /= len(dataset_obj) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()

#----------------------------------------------------------------------------

def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

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

    
class CommaSeparatedList(click.ParamType):
    
    name = 'list'
    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')


def save_image(img, num_channel, fname):
    assert num_channel in [1, 3]
    if num_channel == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if num_channel == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

def save_fid(fid, fname):        
    formatted_string = f'{fid:g}'  # Format the number using :g specifier
    # Open the file in write mode and save the formatted string
    with open(fname, 'w') as file:
        file.write(formatted_string)  
        
#----------------------------------------------------------------------------

@click.command()

@click.option('--init_sigma',    help='Noise standard deviation that is fixed during distillation and generation', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=2.5, show_default=True)
@click.option('--data_stat',     help='Path to the dataset stats', metavar='ZIP|DIR',               type=str, default=None)
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--detector', 'detector_url',  help='detector pickle filename', metavar='PATH|URL',                      type=str, default='https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl')
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--ref', 'ref_path',      help='Dataset reference statistics ', metavar='NPZ|URL',    type=str, required=True)


def main(**kwargs):
    """Generate random images using the techniques described in the paper
    "Score identity Distillation: Exponentially Fast Distillation of Pretrained Diffusion Models for One-Step Generation".

    Examples:

    \b
    # Generate 50000 images, save them as out/*.png, and then compute the FID by comparing the statistics of the generated images and the statistics of the reference data
    
    #Cifar10 unconditional, alpha=1.2
    python sid_generate.py --outdir=image_experiment/out --seeds=0-49999 --batch=128 --network='https://huggingface.co/UT-Austin-PML/SiD/resolve/main/cifar10-uncond/alpha1.2/network-snapshot-1.200000-403968.pkl' --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
    
    #Cifar10 conditional, alpha=1.2
    python sid_generate.py --outdir=image_experiment/out --seeds=0-49999 --batch=128 --network='https://huggingface.co/UT-Austin-PML/SiD/resolve/main/cifar10-cond/alpha1.2/network-snapshot-1.200000-713312.pkl' --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
    

    \b
    # Generate 50000 images using 4 GPUs
    
    #ImageNet 64x64, alpha=1.2
    torchrun --standalone --nproc_per_node=4 sid_generate.py --outdir=out --seeds=0-49999 --batch=128 --network='https://huggingface.co/UT-Austin-PML/SiD/resolve/main/imagenet64/alpha1.2/network-snapshot-1.200000-939176.pkl' --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz
    
    #FFHQ 64x64, alpha=1.2
    torchrun --standalone --nproc_per_node=4 sid_generate.py --outdir=out --seeds=0-49999 --batch=128 --network='https://huggingface.co/UT-Austin-PML/SiD/resolve/main/ffhq64/alpha1.2/network-snapshot-1.200000-498176.pkl' --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz
    
    #AFHQ-v2 64x64, alpha=1
    torchrun --standalone --nproc_per_node=4 sid_generate.py --outdir=out --seeds=0-49999 --batch=128 --network='https://huggingface.co/UT-Austin-PML/SiD/resolve/main/afhq64/alpha1/network-snapshot-1.000000-371712.pkl' --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz
  
    
    """
    
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    
    
    network_pkl=opts.network_pkl
    
    
    seeds = opts.seeds
    max_batch_size = opts.max_batch_size
    # image_outdir = opts.image_outdir
    
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    device = torch.device('cuda')

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')    
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

        
 
        
    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{opts.outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
        if opts.class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, opts.class_idx] = 1

        # Generate images.
        sampler_kwargs = {}
        images = sid_sampler(net, latents, class_labels, randn_like=rnd.randn_like,init_sigma=opts.init_sigma, **sampler_kwargs)

        # Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        image_dir = opts.outdir
        os.makedirs(image_dir, exist_ok=True)
        for seed, image_np in zip(batch_seeds, images_np):
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            save_image(img=image_np,num_channel=image_np.shape[2],fname=image_path)

    # Done.
    torch.distributed.barrier()
    dist.print0('Done Generating images.')

    dist.print0('Computing FID')
    #batch = opts.max_batch_size
    image_path = opts.outdir
    ref_path = opts.ref_path
    detector_url=opts.detector_url
    num_expected = len(seeds)
    seed = 0

    dist.print0(f'Loading dataset reference statistics from "{ref_path}"...')
    ref = None
    if dist.get_rank() == 0:
        with dnnlib.util.open_url(ref_path) as f:
            ref = dict(np.load(f))

    mu, sigma = calculate_inception_stats(detector_url=detector_url,image_path=image_path, num_expected=num_expected, seed=seed, max_batch_size=max_batch_size)
    dist.print0('Calculating FID...')
    if dist.get_rank() == 0:
        fid = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
        print(f'{fid:g}')
        
        match = re.search(r"-(\d+)\.pkl$", network_pkl)
        if match:
            # If a match is found, extract the number part
            number_part = match.group(1)
        else:
            # If no match is found, handle the situation (e.g., use a default value or skip processing)
            number_part = '_final'  # Or any other handling logic you prefer
        
        txt_path = os.path.join(os.path.dirname(opts.outdir),f'fid{number_part}.txt')
        save_fid(fid=fid,fname=txt_path)
        # os.remove(image_path)
    torch.distributed.barrier()
    dist.print0('Done Computing FID')
    
#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------