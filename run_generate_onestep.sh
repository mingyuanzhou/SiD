#!/bin/bash

network=$1

#sh run_generate_onestep.sh 'image_experiment/sid-train-runs/cifar10-uncond/00001-cifar10-32x32-uncond-ddpmpp-edmglr1e-05-lr1e-05-initsigma2.5-gpus4-alpha1.2-batch256-tmax800-fp32/network-snapshot-1.200000-001024.pkl' 

python generate_onestep.py --outdir='image_experiment/sid-train-runs/out' --seeds=0-63 --batch=64  --network="$network"