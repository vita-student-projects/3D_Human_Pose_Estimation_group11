#!/bin/bash
#conda activate dlav

python train.py --dataset_path '../data/S11/S_11_C_4_1.h5' --epochs 1 --batch_size 1 --num_workers 1 --lr 0.1 --save_interval 1 --checkpoint_dir 'checkpoints'

