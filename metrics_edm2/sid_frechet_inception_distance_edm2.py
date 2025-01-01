
# This file has been modified from the original located at:
# https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/metrics/frechet_inception_distance.py

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

import os
import numpy as np
import scipy.linalg
#from . import sid_metric_utils as metric_utils
from . import sid_metric_utils_edm2 as metric_utils


#----------------------------------------------------------------------------

def compute_fid(opts, max_real, num_gen,batch_size=64,batch_gen=1):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    #detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    if opts.detector_url is None:
        detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    else:
        detector_url = opts.detector_url
    
    detector_dino = getattr(opts, 'detector_dino', None)
    
    
    opts.pr_flag = False # Set to False when using metrics other than precision/recall.
    
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    if detector_dino is None:
        opts.dino_flag=False
        # Add an option to utilize precomputed statistics for reference data.
        if opts.data_stat is not None:
            mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real)

        else:
            mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real).get_mean_cov()

        # The generator component remains unchanged from the original code.
        mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, batch_size=batch_size,batch_gen=batch_gen,capture_mean_cov=True, max_items=num_gen).get_mean_cov()

        if opts.rank != 0:
            return float('nan')

        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        return float(fid)
    else:
        opts.dino_flag=True
        
        if opts.data_stat is not None:
            mu_real, sigma_real, mu_real_dino, sigma_real_dino = metric_utils.compute_feature_stats_for_dataset(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real)
        else:
            mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real).get_mean_cov()
        print('mu_real_dino:',mu_real_dino)
        print('sigma_real_dino:',sigma_real_dino)
        # The generator component remains unchanged from the original code.
        fid_stats, dino_stats = metric_utils.compute_feature_stats_for_generator(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, batch_size=batch_size,batch_gen=batch_gen,capture_mean_cov=True, max_items=num_gen)
        mu_gen, sigma_gen=fid_stats.get_mean_cov()
        mu_gen_dino, sigma_gen_dino=dino_stats.get_mean_cov()
        
        if opts.rank != 0:
            return float('nan'), float('nan')
        
        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        print('fid:',fid)
        
        #print('dino_stats:',dino_stats)
        
        print('mu_gen_dino:',mu_gen_dino)
        print('sigma_gen_dino:',sigma_gen_dino)
        

        m = np.square(mu_gen_dino - mu_real_dino).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen_dino, sigma_real_dino), disp=False) # pylint: disable=no-member
        fid_dino = np.real(m + np.trace(sigma_gen_dino + sigma_real_dino - s * 2))
        print('fid_dino:',fid_dino)
        return float(fid), float(fid_dino)

#----------------------------------------------------------------------------