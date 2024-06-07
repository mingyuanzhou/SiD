# Copyright (c) 2024, Mingyuan Zhou. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt

"""Distill pretraind diffusion-based generative model using the techniques described in the
paper "Score identity Distillation: Exponentially Fast Distillation of
Pretrained Diffusion Models for One-Step Generation"."""

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc

from metrics import sid_metric_main as metric_main


#----------------------------------------------------------------------------
def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)


#----------------------------------------------------------------------------
# Helper methods

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

def save_image(img, num_channel, fname):
    assert C in [1, 3]
    if num_channel == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if num_channel == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)
        

def save_data(data, fname):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

def save_pt(pt, fname):
    torch.save(pt, fname)


def calculate_metric(metric,  G, init_sigma, dataset_kwargs, num_gpus, rank, local_rank, device):
    return metric_main.calc_metric(metric=metric,G=G, init_sigma=init_sigma,
        dataset_kwargs=dataset_kwargs, num_gpus=num_gpus, rank=rank, local_rank=local_rank, device=device)

def append_line(jsonl_line, fname):
    with open(fname, 'at') as f:
        f.write(jsonl_line + '\n')


#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    fake_score_optimizer_kwargs   = {},       # Options for fake score network optimizer.
    g_optimizer_kwargs    = {},     # Options for generator optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    #
    loss_scaling_G      = 100,       # Loss scaling factor of G for reducing FP16 under/overflows.
    #
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot for initialization, None = random initialization.
    resume_training     = None,     # Resume training from the given network snapshot.
    resume_kimg         = 0,        # Start from the given training progress.
    alpha               = 1,         # loss = L2-alpha*L1
    tmax                = 800,        #We add noise at steps 0 to tmax, tmax <= 1000
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    metrics             = None,
    init_sigma          = None,
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)

    #Construct the pretrained (true) score network f_phi
    true_score = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    true_score.eval().requires_grad_(False).to(device)

    #Construct the generator (fake) score network f_psi
    fake_score = copy.deepcopy(true_score).train().requires_grad_(True).to(device)

    #Construct the generator G_theta
    G = copy.deepcopy(true_score).train().requires_grad_(True).to(device)
     
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, G.img_channels, G.img_resolution, G.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, G.label_dim], device=device)
            misc.print_module_summary(G, [images, sigma, labels], max_nesting=2)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    fake_score_optimizer = dnnlib.util.construct_class_by_name(params=fake_score.parameters(), **fake_score_optimizer_kwargs) # subclass of torch.optim.Optimizer
    g_optimizer = dnnlib.util.construct_class_by_name(params=G.parameters(), **g_optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    
    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from URL "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        
        dist.print0('Loading network completed')
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=true_score, require_all=False)
        
        if resume_training is not None:
            dist.print0('checkpoint path:',resume_training)
            data = torch.load(resume_training, map_location=torch.device('cpu'))
            misc.copy_params_and_buffers(src_module=data['fake_score'], dst_module=fake_score, require_all=True)
            misc.copy_params_and_buffers(src_module=data['G'], dst_module=G, require_all=True)
            G_ema = copy.deepcopy(G).eval().requires_grad_(False)
            misc.copy_params_and_buffers(src_module=data['G_ema'], dst_module=G_ema, require_all=True)
            G_ema.eval().requires_grad_(False)
            fake_score_optimizer.load_state_dict(data['fake_score_optimizer_state'])
            g_optimizer.load_state_dict(data['g_optimizer_state'])
            del data # conserve memory
            dist.print0('Loading checkpoint completed')
            if dist.get_rank() == 0:
                os.remove(resume_training) 
            dist.print0('Setting up optimizer...')
            fake_score_ddp = torch.nn.parallel.DistributedDataParallel(fake_score, device_ids=[device], broadcast_buffers=False,find_unused_parameters=False)
            G_ddp = torch.nn.parallel.DistributedDataParallel(G, device_ids=[device], broadcast_buffers=False,find_unused_parameters=False)

        else:     
            # Setup optimizer.
            misc.copy_params_and_buffers(src_module=data['ema'], dst_module=fake_score, require_all=False)
            misc.copy_params_and_buffers(src_module=data['ema'], dst_module=G, require_all=False)
            dist.print0('Setting up optimizer...')
            fake_score_ddp = torch.nn.parallel.DistributedDataParallel(fake_score, device_ids=[device], broadcast_buffers=False,find_unused_parameters=False)
            G_ddp = torch.nn.parallel.DistributedDataParallel(G, device_ids=[device], broadcast_buffers=False,find_unused_parameters=False)
            G_ema = copy.deepcopy(G).eval().requires_grad_(False)
            misc.copy_params_and_buffers(src_module=data['ema'], dst_module=G_ema, require_all=False)
            del data # conserve memory
        fake_score_ddp.eval().requires_grad_(False)
        G_ddp.eval().requires_grad_(False)
        
        
    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    
    
    
    if dist.get_rank() == 0:
        grid_size, images, labels = setup_snapshot_image_grid(training_set=dataset_obj)
        grid_z = init_sigma*torch.randn([labels.shape[0], G_ema.img_channels, G_ema.img_resolution, G_ema.img_resolution], device=device)
        grid_z = grid_z.split(batch_gpu)

        grid_c = torch.from_numpy(labels).to(device)
        grid_c = grid_c.split(batch_gpu)
        if resume_training is None:
            print('Exporting sample images...')
            save_image_grid(img=images, fname=os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
            images = torch.cat([G_ema(z, (init_sigma*torch.ones(z.shape[0],1,1,1)).to(z.device), c, augment_labels=torch.zeros(z.shape[0], 9).to(z.device)).cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            save_image_grid(img=images, fname=os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)
            del images

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    stats_metrics = dict()
    
    data = dict(ema=G_ema)
    for key, value in data.items():
        if isinstance(value, torch.nn.Module):
            value = copy.deepcopy(value).eval().requires_grad_(False)
            # misc.check_ddp_consistency(value)
            data[key] = value.cpu()
        del value # conserve memory
        
    if dist.get_rank() == 0:
        save_data(data=data, fname=os.path.join(run_dir, f'network-snapshot-{alpha:03f}-{cur_nimg//1000:06d}.pkl'))
    
    del data # conserve memory
    dist.print0('Exporting sample images...')

    
    if resume_training is None: 
        if dist.get_rank() == 0:
            images = torch.cat([G_ema(z, init_sigma*torch.ones(z.shape[0],1,1,1).to(z.device).to(z.dtype), c, augment_labels=torch.zeros(z.shape[0], 9).to(z.device).to(z.dtype)).cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            save_image_grid(img=images, fname=os.path.join(run_dir, f'fakes_{alpha:03f}_{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)
            del images
        
        dist.print0('Evaluating metrics...')

        for metric in metrics:
            result_dict = calculate_metric(metric=metric, G=G_ema, init_sigma=init_sigma,
                dataset_kwargs=dataset_kwargs, num_gpus=dist.get_world_size(), rank=dist.get_rank(), local_rank=dist.get_local_rank(), device=device)
            if dist.get_rank() == 0:
                print(result_dict.results)
                metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=f'fakes_{alpha:03f}_{cur_nimg//1000:06d}.png', alpha=alpha)          
            stats_metrics.update(result_dict.results)
      
    while True:        
        
        #Update fake score network f_psi
        # Accumulate gradients.
        fake_score_ddp.train().requires_grad_(True)
        fake_score_optimizer.zero_grad(set_to_none=True)

        for round_idx in range(num_accumulation_rounds):
            images, labels = next(dataset_iterator)
            images = images.to(device).to(torch.float32) / 127.5 - 1
            labels = labels.to(device)
            z = init_sigma*torch.randn_like(images)
            with misc.ddp_sync(G_ddp, False):
                images = G_ddp(z, init_sigma*torch.ones(z.shape[0],1,1,1).to(z.device), labels, augment_labels=torch.zeros(z.shape[0], 9).to(z.device))
            with misc.ddp_sync(fake_score_ddp, (round_idx == num_accumulation_rounds - 1)):
                loss = loss_fn(fake_score=fake_score_ddp, images=images, labels=labels, augment_pipe=augment_pipe) 
                loss=loss.sum().mul(loss_scaling / batch_gpu_total)
                loss.backward()
        loss_fake_score_print = loss.item()
        training_stats.report('fake_score_Loss/loss', loss_fake_score_print)

        fake_score_ddp.eval().requires_grad_(False)

        for param in fake_score.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

        fake_score_optimizer.step()

        #Update generator G_theta
        G_ddp.train().requires_grad_(True)
        g_optimizer.zero_grad(set_to_none=True)

        for round_idx in range(num_accumulation_rounds):
            images, labels = next(dataset_iterator)
            images = images.to(device).to(torch.float32) / 127.5 - 1
            labels = labels.to(device)
            z = init_sigma*torch.randn_like(images)
            with misc.ddp_sync(G_ddp, (round_idx == num_accumulation_rounds - 1)):
                images = G_ddp(z, init_sigma*torch.ones(z.shape[0],1,1,1).to(z.device), labels, augment_labels=torch.zeros(z.shape[0], 9).to(z.device))
            with misc.ddp_sync(fake_score_ddp, False):
                loss = loss_fn.generator_loss(true_score=true_score, fake_score=fake_score_ddp, images=images, labels=labels, augment_pipe=None,alpha=alpha,tmax=tmax)
                loss=loss.sum().mul(loss_scaling_G / batch_gpu_total)
                loss.backward()
        lossG_print = loss.item()
        training_stats.report('G_Loss/loss', lossG_print)

        G_ddp.eval().requires_grad_(False)

        for param in G.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

        g_optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
                
        for p_ema, p_true_score in zip(G_ema.parameters(), G.parameters()):
            p_ema.copy_(p_true_score.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        fields += [f"loss_fake_score {training_stats.report0('fake_score_Loss/loss', loss_fake_score_print):<6.2f}"]
        fields += [f"loss_G {training_stats.report0('G_Loss/loss', lossG_print):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')
                        
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0 or cur_tick in [10,20,30,40,50,60,70,80,90,100]):

            dist.print0('Exporting sample images...')
            if dist.get_rank() == 0:
                
                images = torch.cat([G_ema(z, init_sigma*torch.ones(z.shape[0],1,1,1).to(z.device).to(z.dtype), c, augment_labels=torch.zeros(z.shape[0], 9).to(z.device).to(z.dtype)).cpu() for z, c in zip(grid_z, grid_c)]).numpy()
                save_image_grid(img=images, fname=os.path.join(run_dir, f'fakes_{alpha:03f}_{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)
                del images
                
            dist.print0('Evaluating metrics...')
            for metric in metrics:
                result_dict = calculate_metric(metric=metric, G=G_ema, init_sigma=init_sigma,
                    dataset_kwargs=dataset_kwargs, num_gpus=dist.get_world_size(), rank=dist.get_rank(), local_rank=dist.get_local_rank(), device=device)
                if dist.get_rank() == 0:
                    print(result_dict.results)
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=f'fakes_{alpha:03f}_{cur_nimg//1000:06d}.png', alpha=alpha)  
                stats_metrics.update(result_dict.results)
                
            data = dict(ema=G_ema)
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    data[key] = value.cpu()
                del value # conserve memory
                
            if dist.get_rank() == 0:
                save_data(data=data, fname=os.path.join(run_dir, f'network-snapshot-{alpha:03f}-{cur_nimg//1000:06d}.pkl'))               
            del data # conserve memory

        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            dist.print0(f'saving checkpoint: training-state-{cur_nimg//1000:06d}.pt')
            save_pt(pt=dict(fake_score=fake_score, G=G, G_ema=G_ema, fake_score_optimizer_state=fake_score_optimizer.state_dict(), g_optimizer_state=g_optimizer.state_dict()), fname=os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                append_line(jsonl_line=json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n', fname=os.path.join(run_dir, f'stats_{alpha:03f}.jsonl'))

        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
