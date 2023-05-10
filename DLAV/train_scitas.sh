#!/bin/bash
#conda activate dlav

python train_scitas.py --dataset_path '../data/h3.6' --epochs 1 --batch_size 1 --num_workers 1 --lr 0.1 --save_interval 1 --checkpoint_dir 'checkpoints'