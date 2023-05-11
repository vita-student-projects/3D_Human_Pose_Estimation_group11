#!/bin/bash
#conda activate dlav

python train.py --dataset_path '../data/S11/S_11_C_4_1.h5' --epochs 100 --batch_size 100 --num_workers 1 --lr 0.001 --save_interval 1 --checkpoint_dir 'checkpoints' --data_ratio 1 --validation_ratio 0.3 --test_ratio 0.0