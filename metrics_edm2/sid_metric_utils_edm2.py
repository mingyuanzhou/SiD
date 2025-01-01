# This file has been modified from the original located at:
# https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/metrics/metric_utils.py
# When opts.data_stat is not None and opts.pr_flag is True, the script uses a precomputed NPZ file from the dataset to calculate precision-recall. This is necessary for assessing the precision and recall of a one-step generator distilled from the EDM model pretrained on ImageNet 64x64.

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import hashlib
import pickle
import copy
import uuid
import numpy as np
import torch
import dnnlib
from torch.utils.data import Dataset

from sida_training_edm2.sida_networks_edm2 import generate_multistep


#----------------------------------------------------------------------------
class NumpyArrayDataset(Dataset):
    """Custom Dataset for loading numpy arrays with a placeholder for labels."""
    
    def __init__(self, np_array):
        """
        Args:
            np_array (numpy.ndarray): A numpy array containing the data.
        """
        # Assuming the input np_array is in shape (B, H, W, C), we transpose it to (B, C, H, W) for PyTorch
        self.data = torch.from_numpy(np_array).float().permute(0, 3, 1, 2)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Returning a dummy label here, replace or modify as necessary based on your dataset
        dummy_label = 0
        return self.data[idx], dummy_label

#----------------------------------------------------------------------------

class MetricOptions:
    def __init__(self, G=None, vae=None, scale=None, bias=None, init_sigma=None, G_kwargs={}, dataset_kwargs={}, data_stat=None, detector_url=None,num_gpus=1, rank=0, local_rank=0, device=None, progress=None, cache=True,       dtype=torch.float16,train_sampler=False,num_steps=1,batch_size=64,batch_gen=1,detector_dino=None):
        assert 0 <= rank < num_gpus
        
        self.G              = G
        
        self.vae = vae
        self.scale=scale
        self.bias=bias
        
        self.dtype=dtype
        self.train_sampler=train_sampler
        self.num_steps=num_steps
        
        self.G_kwargs       = dnnlib.EasyDict(G_kwargs)
        self.init_sigma = init_sigma
        self.dataset_kwargs = dnnlib.EasyDict(dataset_kwargs) 
        self.data_stat      = data_stat
        self.detector_url   = detector_url 
        self.num_gpus       = num_gpus
        self.rank           = rank
        self.local_rank     = local_rank
        self.device         = device if device is not None else torch.device('cuda', rank)
        self.progress       = progress.sub() if progress is not None and rank == 0 else ProgressMonitor()
        self.cache          = cache
        self.batch_size = batch_size
        self.batch_gen = batch_gen
        self.detector_dino   = detector_dino 

#----------------------------------------------------------------------------

_feature_detector_cache = dict()

def get_feature_detector_name(url):
    return os.path.splitext(url.split('/')[-1])[0]

def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier() # leader goes first
        with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
            _feature_detector_cache[key] = torch.jit.load(f).eval().to(device)
        if is_leader and num_gpus > 1:
            torch.distributed.barrier() # others follow
    return _feature_detector_cache[key]

#----------------------------------------------------------------------------

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj
    
class FeatureStats_dino:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats_dino(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj

#----------------------------------------------------------------------------

class ProgressMonitor:
    def __init__(self, tag=None, num_items=None, flush_interval=1000, verbose=False, progress_fn=None, pfn_lo=0, pfn_hi=1000, pfn_total=1000):
        self.tag = tag
        self.num_items = num_items
        self.verbose = verbose
        self.flush_interval = flush_interval
        self.progress_fn = progress_fn
        self.pfn_lo = pfn_lo
        self.pfn_hi = pfn_hi
        self.pfn_total = pfn_total
        self.start_time = time.time()
        self.batch_time = self.start_time
        self.batch_items = 0
        if self.progress_fn is not None:
            self.progress_fn(self.pfn_lo, self.pfn_total)

    def update(self, cur_items):
        assert (self.num_items is None) or (cur_items <= self.num_items)
        if (cur_items < self.batch_items + self.flush_interval) and (self.num_items is None or cur_items < self.num_items):
            return
        cur_time = time.time()
        total_time = cur_time - self.start_time
        time_per_item = (cur_time - self.batch_time) / max(cur_items - self.batch_items, 1)
        if (self.verbose) and (self.tag is not None):
            print(f'{self.tag:<19s} items {cur_items:<7d} time {dnnlib.util.format_time(total_time):<12s} ms/item {time_per_item*1e3:.2f}')
        self.batch_time = cur_time
        self.batch_items = cur_items

        if (self.progress_fn is not None) and (self.num_items is not None):
            self.progress_fn(self.pfn_lo + (self.pfn_hi - self.pfn_lo) * (cur_items / self.num_items), self.pfn_total)

    def sub(self, tag=None, num_items=None, flush_interval=1000, rel_lo=0, rel_hi=1):
        return ProgressMonitor(
            tag             = tag,
            num_items       = num_items,
            flush_interval  = flush_interval,
            verbose         = self.verbose,
            progress_fn     = self.progress_fn,
            pfn_lo          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_lo,
            pfn_hi          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_hi,
            pfn_total       = self.pfn_total,
        )

#----------------------------------------------------------------------------

def compute_feature_stats_for_dataset(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, data_loader_kwargs=None, max_items=None, **stats_kwargs):
    if opts.data_stat is not None:   #use the precomputed dataset stats, which is needed when computing the Precision and Recall for the distill Imagenet-64x64 EDM model
        
        loaded_npz = None
        try:
            # Assuming `open_url` returns a file-like object
            with dnnlib.util.open_url(opts.data_stat) as f:
                # Load the file using numpy.load
                if opts.data_stat.lower().endswith('.npz'): # backwards compatibility with https://github.com/NVlabs/edm
                    loaded_npz = dict(np.load(f))
                else:
                    loaded_results=pickle.load(f)
                    loaded_npz=loaded_results['fid']
                    if opts.dino_flag:
                        loaded_dino_npz = loaded_results['fd_dinov2']
                
                #print("Data loaded successfully", loaded_npz)
        except Exception as e:
            print("Failed to load the data:", str(e))
        
        #loaded_npz = np.load(opts.data_stat)
        
        if opts.pr_flag:  #return the stats needed for computing Precision and Recall
            dataset = NumpyArrayDataset(loaded_npz['arr_0'])
            dataset.name = 'precomputed-feature'
        else: #return the stats needed for computing FID
            opts.dino_flag = getattr(opts, 'dino_flag', False)
            if not opts.dino_flag:
                return loaded_npz['mu'], loaded_npz['sigma']
            else:
                #loaded_dino_npz=pickle.load(f)['fd_dinov2']
                return loaded_npz['mu'], loaded_npz['sigma'], loaded_dino_npz['mu'], loaded_dino_npz['sigma']
    else: #compute the dataset feature stats using the training data
        dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    # Try to lookup from cache.
    cache_file = None
    if opts.cache:
        # Choose cache file name.
        args = dict(dataset_kwargs=opts.dataset_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs)
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        cache_tag = f'{dataset.name}-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        cache_file = dnnlib.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file) if opts.rank == 0 else False
        if opts.num_gpus > 1:
            flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
            torch.distributed.broadcast(tensor=flag, src=0)
            flag = (float(flag.cpu()) != 0)

        # Load.
        if flag:
            return FeatureStats.load(cache_file)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    for images, _labels in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images.to(opts.device), **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)

    # Save to cache.
    if cache_file is not None and opts.local_rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file) # atomic
    return stats

#----------------------------------------------------------------------------

def compute_feature_stats_for_generator(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, batch_gen=None, jit=False,**stats_kwargs):
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # Setup generator and load labels.
    #G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    
    #opts.dino_flag = getattr(opts, 'dino_flag', False)
    #if opts.dino_flag:
    
    #detector_dino_url = opts.detector_dino_url
    #if detector_dino_url is not None:
    # if detector_dino is not None:
    #     dino_flag = True
    # else:
    #     dino_flag = False
    # #print('dino_url0:', opts.detector_dino_url)
    # print('dino_flag0:', dino_flag)
    
    G = opts.G
    #print(1.0)
    init_sigma = opts.init_sigma
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)

#     # Image generation func.
#     def run_generator(z, c, init_sigma):
#         with torch.no_grad():
#             img = G(z, init_sigma*torch.ones(z.shape[0],1,1,1).to(z.device), c, augment_labels=torch.zeros(z.shape[0], 9).to(z.device))
#         img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
#         return img
    #print(2.0)
    def run_generator(z, c, init_sigma, opts):
        with torch.no_grad():
            # Ensure all necessary attributes are in opts and are of tensor type
            scale = torch.tensor(opts.scale, dtype=torch.float32, device=z.device) if hasattr(opts, 'scale') else torch.tensor(1.0, device=z.device)
            bias = torch.tensor(opts.bias, dtype=torch.float32, device=z.device) if hasattr(opts, 'bias') else torch.tensor(0.0, device=z.device)
            vae = opts.vae if hasattr(opts, 'vae') else None

            # Initialize sigma tensor and pass through G
            init_sigma_tensor = init_sigma * torch.ones(z.shape[0], 1, 1, 1, device=z.device)
            
            if opts.num_steps==1:
                img = G(z, init_sigma_tensor, c, augment_labels=torch.zeros(z.shape[0], 9, device=z.device))
            else:
                img =generate_multistep(G,z, init_sigma_tensor, c, num_steps=opts.num_steps)

            if vae is not None:
                vae.to(z.device)  # Ensure VAE is on the same device as the input tensor
                #img = img.to(torch.float32)
                img = (img - bias.reshape(1, -1, 1, 1)) / scale.reshape(1, -1, 1, 1)
                img = vae.decode(img).sample  # Ensure `.sample` is compatible with `vae.decode`

            # Final scaling to uint8 for image representation
            img = img.clamp(0, 1).mul(255).to(torch.uint8)
            #print(img.shape)  # For debugging purposes

        return img
    
#     def run_generator(z, c, init_sigma,opts):
#         with torch.no_grad():
#             init_sigma_tensor = init_sigma * torch.ones(z.shape[0], 1, 1, 1).to(z.device)
#             img = G(z, init_sigma_tensor, c,augment_labels=torch.zeros(z.shape[0], 9).to(z.device))
            
#             scale = torch.tensor(opts.scale, dtype=torch.float32, device=x.device)
#             bias = torch.tensor(opts.bias, dtype=torch.float32, device=x.device)
#             vae=opts.vae
#             vae.to(x.device)  # Ensure VAE is on the same device as the input tensor
#             x = x.to(torch.float32)
#             x = x - bias.reshape(1, -1, 1, 1)
#             x = x / scale.reshape(1, -1, 1, 1)
#             x = vae.decode(x).sample  # Ensure `vae.decode(x)` returns a compatible object with `.sample`
#             #x = x.clamp(0, 1).mul(255).to(torch.uint8)
#             #return x
            
#             img=x
#             #img = opts.vae(opts. img)
#             img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
#             print(img.shape)
#         return img

    # JIT.
    if jit:
        z = init_sigma*torch.zeros([batch_gen, G.img_channels, G.img_resolution, G.img_resolution], device=opts.device)
        c = torch.zeros([batch_gen, G.c_dim], device=opts.device)
        run_generator = torch.jit.trace(run_generator, [z, c, init_sigma], check_trace=False)
    #print(3.0)
    # Initialize.
    stats = FeatureStats(**stats_kwargs)
    #if dino_flag:
    
    detector_dino = getattr(opts, 'detector_dino', None)
    
    if detector_dino is not None:
        stats_dino = FeatureStats_dino(**stats_kwargs)
    
    assert stats.max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    #print(4.0)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)
#     if opts.dino_flag:
#         detector_dino = vit_large(
#             patch_size=14,
#             img_size=526,
#             init_values=1.0,
#             block_chunks=0
#          )
        
        
#         # detector_dino_state_dict = get_feature_detector(url=detector_dino_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)
#         # detector_dino.load_state_dict(detector_dino_state_dict)
#         # detector_dino.to(opts.device)
        
#         detector_dino.load_state_dict(torch.load(detector_dino_url,map_location=opts.device))
#         detector_dino.to(opts.device)
        
#         #detector_dino.to(opts.device)
#         #detector_dino.load_state_dict(get_feature_detector(url=detector_dino_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose))
        

    
    print(f'{opts.rank},5.0')
    # Main loop.
    while not stats.is_full():
        images = []
        for _i in range(batch_size // batch_gen):
            #print(f'{opts.rank},5.0')
            if hasattr(G, 'img_channels'):
                img_channels = G.img_channels
                img_resolution=G.img_resolution
            else:
                img_channels = 4
                img_resolution=64
            
            
            z = init_sigma*torch.randn([batch_gen, img_channels, img_resolution, img_resolution], device=opts.device) 
            c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(batch_gen)]
            c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
            
#             print(z.shape)
#             print(c)
#             print(run_generator(z, c, init_sigma,opts))
            
            images.append(run_generator(z, c, init_sigma,opts))
        images = torch.cat(images)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        with torch.no_grad():
            features = detector(images, **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        if detector_dino is not None:
            x=images
            resize_mode = 'torch'
                # Resize images.
            if resize_mode == 'pil': # Slow reference implementation that matches the original dgm-eval codebase exactly.
                device = x.device
                x = x.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                x = np.stack([np.uint8(PIL.Image.fromarray(xx, 'RGB').resize((224, 224), PIL.Image.Resampling.BICUBIC)) for xx in x])
                x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
            elif resize_mode == 'torch': # Fast practical implementation that yields almost the same results.
                x = torch.nn.functional.interpolate(x.to(torch.float32), size=(224, 224), mode='bicubic', antialias=True)
            else:
                raise ValueError(f'Invalid resize mode "{self.resize_mode}"')
            
            # Adjust dynamic range.
            #x = x.to(torch.float32) / 255
            #x = x - misc.const_like(x, [0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
            #x = x / misc.const_like(x, [0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).reshape(1, -1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).reshape(1, -1, 1, 1)

            x = x.to(torch.float32) / 255
            x = (x - mean.to(x.device)) / std.to(x.device)
            
            # Run DINOv2 model.
            with torch.no_grad():
                features_dino = detector_dino(x)
            stats_dino.append_torch(features_dino, num_gpus=opts.num_gpus, rank=opts.rank)
        
        progress.update(stats.num_items)
    if detector_dino is not None:
         return stats, stats_dino
    else:
        return stats

#----------------------------------------------------------------------------