# Copyright (c) 2024, Mingyuan Zhou. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt


"""Distill pretraind diffusion-based generative model using the techniques described in the
paper "Score identity Distillation: Exponentially Fast Distillation of
Pretrained Diffusion Models for One-Step Generation"."""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import socket
import re
import json
import click
import torch
import dnnlib


import time
import copy
import pickle
import psutil
import PIL.Image
import numpy as np



from torch_utils import distributed as dist

from torch_utils import training_stats
from torch_utils import misc

from metrics import sid_metric_main as metric_main

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

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
    
def calculate_metric(metric, G, init_sigma, dataset_kwargs, num_gpus, rank, device,data_stat):
    return metric_main.calc_metric(metric=metric, G=G, init_sigma=init_sigma,
        dataset_kwargs=dataset_kwargs, num_gpus=num_gpus, rank=rank, device=device,data_stat=data_stat)


def append_line(jsonl_line, fname):
    with open(fname, 'at') as f:
        f.write(jsonl_line + '\n')
        

def save_metric(result_dict, fname):        
    # formatted_string = f'{metric:g}'  # Format the number using :g specifier
    # # Open the file in write mode and save the formatted string
    # with open(fname, 'w') as file:
    #     file.write(formatted_string)  
    with open(fname, "w") as file:
    #with open(fname, 'at') as file:
        for key, value in result_dict.items():
            file.write(f"{key}: {value}\n")



#----------------------------------------------------------------------------


@click.command()
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--data_stat',     help='Path to the dataset stats', metavar='ZIP|DIR',               type=str, default=None)
@click.option('--init_sigma',    help='Noise standard deviation that is fixed during distillation and generation', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=2.5, show_default=True)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)

# Main options.gpu
# Adapted from Diff-Instruct
@click.option('--metrics',       help='Comma-separated list or "none" [default: fid50k_full]',      type=CommaSeparatedList())
# FID metric PT path
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',      help='Path to store the output text file',  type=str, default='evalout')

def main(**kwargs):
    """Distill pretraind diffusion-based generative model using the techniques described in the
paper "Score identity Distillation: Exponentially Fast Distillation of
Pretrained Diffusion Models for One-Step Generation".
    
    Examples:

    \b
    # Distill EDM model for class-conditional CIFAR-10 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 sid_train.py --outdir=training-runs \\
        --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp
    """
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()


    network_pkl=opts.network_pkl
    metrics = opts.metrics
    init_sigma = opts.init_sigma
    data_stat = opts.data_stat

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    device = torch.device('cuda')

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        G_ema = pickle.load(f)['ema'].to(device)
    
    dist.print0(f'Finished loading "{network_pkl}"...')
    
    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    
    dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=opts.cond)
    dataset_kwargs.resolution = G_ema.img_resolution
    dataset_kwargs.max_size=2000000
    dist.print0(dataset_kwargs)
        
    
    match = re.search(r"-(\d+)\.pkl$", network_pkl)
    if match:
        # If a match is found, extract the number part
        number_part = match.group(1)
    else:
        # If no match is found, handle the situation (e.g., use a default value or skip processing)
        number_part = '_final'  # Or any other handling logic you prefer
    for i in range(10):    
        for metric in metrics:
            dist.print0(metric)
            result_dict = calculate_metric(metric=metric, G=G_ema, init_sigma=init_sigma,
                dataset_kwargs=dataset_kwargs, num_gpus=dist.get_world_size(), rank=dist.get_rank(), device=device,data_stat=data_stat)
            if dist.get_rank() == 0:
                print(result_dict.results)
                txt_path = os.path.join(os.path.dirname(opts.outdir),f'{metric}{number_part}_{i}.txt')
                print(txt_path)
                save_metric(result_dict=result_dict,fname=txt_path)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
