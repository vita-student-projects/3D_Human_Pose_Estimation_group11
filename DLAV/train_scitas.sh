#!/bin/bash
#conda activate dlav

python train_scitas.py --dataset_path '../DATA_TEST/dataset.h5' --epochs 1 --batch_size 1 --num_workers 1 --lr 0.1 --save_interval 1 --checkpoint_dir 'checkpoints' --data_ratio 0.02 --validation_ratio 0.3 --test_ratio 0.0