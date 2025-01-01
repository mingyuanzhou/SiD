# Copyright (c) 2024, Mingyuan Zhou. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt

"""Distill pretraind diffusion-based generative model using the techniques described in the
paper "Adversarial Score Identity Distillation: Rapidly Surpassing the Teacher in One Step"."""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import socket
import re
import json
import click
import torch
import dnnlib

from torch_utils import distributed as dist
from sida_training_edm2 import sida_training_loop_edm2 as training_loop

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

from typing import Optional, Any

import shutil


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
    


def find_latest_checkpoint(path: str, dist: Any) -> Optional[str]:
    """Finds the latest checkpoint and return path, otherwise return None."""
    if path is None:
        print("Path is None, returning None.")
        return None  # Fixed from returning path to returning None directly

    latest_file = None
    latest_number = -1
    print(f'Finding latest checkpoint in {path}')
   
    if os.path.isdir(path):  
        for root, _, files in os.walk(path):
            print(f'Inspecting directory: {root} with {len(files)} files.')
            for file in files:
                print(f'Checking file: {file}')
                if file.startswith('training-state-') and file.endswith('.pt'):
                    # Extract the number from the file name
                    number_part = file[len('training-state-'):-len('.pt')]
                    print(f'Found checkpoint file: {file} with number part: {number_part}')
                    try:
                        number = int(number_part)
                        if number > latest_number:
                            print(f'New latest checkpoint found: {file} with number {number}')
                            latest_number = number
                            latest_file = os.path.join(root, file)
                    except ValueError:
                        print(f'Could not convert {number_part} to an integer, skipping file.')
                        continue
    else:
        print(f'Path is not a directory; assuming path is a file: {path}')
        latest_file = path  

    if latest_file is None:
        print('No latest checkpoint found')
        return None

    print(f'Found latest checkpoint file: {latest_file}')
    return latest_file
    

#----------------------------------------------------------------------------
@click.command()

# Main options.gpu
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=False)
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--data_stat',     help='Path to the dataset stats', metavar='ZIP|DIR',               type=str, default=None)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=True, show_default=True)
@click.option('--arch',          help='Network architecture',       type=str, default='edm2-img512-xs', show_default=True)
@click.option('--precond',       help='Preconditioning & loss function', metavar='vp|ve|edm',       type=click.Choice(['vp', 've', 'edm','edm2']), default='edm2', show_default=True)

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                              type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--augment',       help='Augment probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.12, show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                     type=bool, default=False, show_default=True)

# Performance-related.
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   type=bool, default=False, show_default=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=200, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)
@click.option('--metrics',       help='Comma-separated list or "none" [default: fid50k_full]',      type=CommaSeparatedList())
@click.option('--edm_model',     help='edm_model', type=str)


# Parameters for SiD
@click.option('--init_sigma',    help='Noise standard deviation that is fixed during distillation and generation', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=2.5, show_default=True)
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--lsg',           help='Loss scaling G', metavar='FLOAT',                            type=click.FloatRange(min=0, min_open=True), default=100, show_default=True)
@click.option('--alpha',         help='L2-alpha*L1', metavar='FLOAT',                               type=click.FloatRange(min=-1000, min_open=True), default=1.0, show_default=True)
@click.option('--tmax',          help='the reverse sampling starting step corresoinding to the largest allowed noise level, in [0,1000]', metavar='INT',  type=click.IntRange(min=0), default=800, show_default=True)
@click.option('--lr',            help='Learning rate of fake score estimation network', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=1e-5, show_default=True)
@click.option('--glr',           help='Learning rate of fake data generator', metavar='FLOAT',       type=click.FloatRange(min=0, min_open=True), default=1e-5, show_default=True)
@click.option('--g_beta1',       help='beta_1 of the Adam optimizer for generator', metavar='FLOAT', type=click.FloatRange(min=0, min_open=False), default=0, show_default=True)


@click.option('--detector_url',     help='detector needed for computing FID',default='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt',  type=str)


#Parameters added for SiDA
@click.option('--lsd',           help='Loss scaling GAN discriminator', metavar='FLOAT',             type=click.FloatRange(min=-10, min_open=True), default=1, show_default=True)
@click.option('--lsg_gan',       help='Loss scaling GAN genenerator loss', metavar='FLOAT',          type=click.FloatRange(min=-10, min_open=True), default=.01, show_default=True)
@click.option('--sid_model',     help='Path to the predistilled SiD one-step generator',             type=str,default=None, show_default=True)
@click.option('--use_gan',       help='Whether adding adversarial training into SiD, i.e., run SiDA if it is True and SiD if it is False', metavar='BOOL',   type=bool, default=True, show_default=True)
@click.option('--vae',     help='stabilityai/sd-vae-ft-mse, needed to encode/decode images of resolution 512x512', default='stabilityai/sd-vae-ft-mse', type=str)
@click.option('--return_logvar',       help='Whether using return_logvar', metavar='BOOL',   type=bool, default=True, show_default=True)
@click.option('--force_normalization',       help='Whether using forced normalization', metavar='BOOL',   type=bool, default=True, show_default=True)

@click.option('--train_mode', help='Set to True for distillation and False for evaluation', type=bool, default=True, show_default=True)
@click.option('--num_steps_train',         help='number of generation steps (NFEs) used for training', metavar='INT',   type=int,  default=1,   show_default=True)
@click.option('--num_steps_eval',          help='number of generation steps (NFEs) used for evaluation', metavar='INT',   type=int,  default=1,   show_default=True)

def main(**kwargs):
    """Distill pretraind diffusion-based generative model using the techniques described in the
paper "Adversarial Score Identity Distillation: Rapidly Surpassing the Teacher in One Step".
    
    """
    
    opts = dnnlib.EasyDict(kwargs)
    
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # The context has already been set

    dist.init()
        
    # Initialize config dict.
    c = dnnlib.EasyDict()

    c.pretrained_vae_model_name_or_path=opts.vae
    if opts.use_gan:
        c.dataset_kwargs = dnnlib.EasyDict(class_name='edm2.dataset.ImageFolderDataset', path=opts.data, use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache)
    else:
        c.dataset_kwargs = dnnlib.EasyDict(class_name='sida_training_edm2.json_labels_dataset.ImageDataset', path=opts.data,resolution=64,name='imagenet512',use_labels=opts.cond)
        
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    #c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=32, prefetch_factor=4)
    
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.generator_kwargs= dnnlib.EasyDict()

    c.fake_score_optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.0, 0.999], eps = 1e-8 if not opts.fp16 else 1e-6)
    c.g_optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.glr, betas=[opts.g_beta1, 0.999], eps = 1e-8 if not opts.fp16 else 1e-6)
    
    c.init_sigma = opts.init_sigma

    # Validate dataset options.
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        c.dataset_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
        c.dataset_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
        if opts.cond and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True requires labels specified in dataset.json')
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Network architecture.

    assert opts.arch in {'edm2-img512-xs','edm2-img512-s','edm2-img512-m','edm2-img512-l','edm2-img512-xl','edm2-img512-xxl'}
    opts.channels_g = None
    if opts.arch == 'edm2-img512-xs':
        opts.channels=128
        opts.dropout=0.00
        opts.P_mean=-0.4
        opts.P_std=1.0
        opts.channels_g=128
        opts.dropout_g=0.00
    elif opts.arch == 'edm2-img512-s':
        opts.channels=192
        opts.dropout=0.00
        opts.P_mean=-0.4
        opts.P_std=1.0
        opts.channels_g=192
        opts.dropout_g=0.00
    elif opts.arch == 'edm2-img512-m':
        opts.channels=256
        opts.dropout=0.10
        opts.P_mean=-0.4
        opts.P_std=1.0
        opts.channels_g=256
        opts.dropout_g=0.10
    elif opts.arch == 'edm2-img512-l':
        opts.channels=320
        opts.dropout=0.10
        opts.P_mean=-0.4
        opts.P_std=1.0
        opts.channels_g=320
        opts.dropout_g=0.10
    elif opts.arch == 'edm2-img512-xl':
        opts.channels=384
        opts.dropout=0.10
        opts.P_mean=-0.4
        opts.P_std=1.0
        opts.channels_g=384
        opts.dropout_g=0.10
    else:
        assert opts.arch == 'edm2-img512-xxl'
        opts.channels=448
        opts.dropout=0.10
        opts.P_mean=-0.4
        opts.P_std=1.0
        opts.channels_g=448
        opts.dropout_g=0.10

    c.network_kwargs.update(model_channels=opts.channels, dropout=opts.dropout)

    assert opts.precond == 'edm2'
    
    c.train_mode=opts.train_mode
    c.return_logvar = opts.return_logvar
    c.force_normalization = opts.force_normalization
    
    if c.return_logvar:
        c.network_kwargs.class_name = 'sida_training_edm2.sida_networks_edm2.EDM_2_Precond_EncoderDecoder'
    else:
        c.network_kwargs.class_name = 'sida_training_edm2.sida_networks_edm2.EDM_2_Precond_Generator'
    
    c.loss_kwargs.class_name = 'sida_training_edm2.sida_loss_edm2.SIDA_EDM_2_Loss'
    c.loss_kwargs.update(P_mean=opts.P_mean, P_std=opts.P_std)
    
    
    #c.encoder_kwargs  = dnnlib.EasyDict(class_name='edm2_training.encoders.StabilityVAEEncoder')
    
    c.generator_kwargs.update(model_channels=opts.channels_g, dropout=opts.dropout_g)
    c.generator_kwargs.class_name='sida_training_edm2.sida_networks_edm2.EDM_2_Precond_Generator'
    
    
    
    c.metrics = opts.metrics

    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)
    
    c.update(loss_scaling_G=opts.lsg)
    
    c.alpha = opts.alpha
    c.tmax = opts.tmax
    
    c.data_stat=opts.data_stat

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    
    c.update(loss_scaling_D=opts.lsd, loss_scaling_G_gan=opts.lsg_gan,use_gan=opts.use_gan)    
        
        
    # Description string.
    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    return_log_var_str = 'return_logvar' if c.return_logvar else 'no_return_logvar'
    forcenorm_str = 'force_normalization' if c.force_normalization else 'no_force_normalization'
    if c.use_gan:
        if opts.sid_model is not None:
            algorithm_str = 'SiD-SiDA'
        else:
            algorithm_str = 'SiDA'
    else:
        algorithm_str = 'SiD'
        
    
    
    desc = f'{dataset_name:s}-{algorithm_str:s}-{cond_str:s}-{opts.arch:s}-{opts.precond:s}-glr{opts.glr}-lr{opts.lr}-ls{opts.ls}_lsg{opts.lsg}_lsd{opts.lsd}_lsg_gan{opts.lsg_gan}-initsigma{opts.init_sigma}-gpus{dist.get_world_size():d}-alpha{c.alpha}-batch{c.batch_size:d}-tmax{c.tmax:d}-{dtype_str:s}-{return_log_var_str:s}-{forcenorm_str:s}'
    
    if opts.sid_model is not None:
        c.sid_model = opts.sid_model
        match = re.search(r'(\d+)\.pkl$', c.sid_model)
        desc += f'_{match.group(1)}'
        
    if opts.batch_gpu is not None:
        desc += f'batchgpu{opts.batch_gpu:d}'
        
    if c.return_logvar:
        desc += f'_logvar'
    
    if opts.desc is not None:
        desc += f'{opts.desc}'
    
    desc += '/'
    
    


    if dist.get_rank() != 0:
        torch.distributed.barrier() # rank 0 goes first
    
    if opts.nosubdir:
        c.run_dir = opts.outdir
        if dist.get_rank()== 0 and not os.path.exists(c.run_dir):
            os.makedirs(c.run_dir, exist_ok=True)
    else:
        c.run_dir = os.path.join(opts.outdir, f'{desc}')
        
        if dist.get_rank()== 0 and not os.path.exists(c.run_dir):
            os.makedirs(c.run_dir, exist_ok=True)
        
        if opts.resume is not None:
            opts.resume = os.path.join(opts.resume, f'{desc}')
            
    if dist.get_rank() == 0:
        torch.distributed.barrier() # other ranks follow
    
    
    c.resume_pkl = opts.edm_model
    
    opts.resume = find_latest_checkpoint(opts.resume,dist)
    if opts.resume is not None:
        print(opts.resume)
        c.resume_training = opts.resume
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_kimg = int(match.group(1))
    

    
    c.detector_url=opts.detector_url

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0(f'alpha:                   {c.alpha}')
    dist.print0(f'tmax:                    {c.tmax}')
    dist.print0(f'precision:               {dtype_str}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------