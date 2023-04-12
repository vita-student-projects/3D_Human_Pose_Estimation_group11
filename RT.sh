#!/bin/bash
#conda activate dlav
#./RT.sh
cd inferenceV3
python validate_evl.py --ckpt_path 'ckpt/hemlets_h36m_lastest.pth' --dataset_path 'data/S11/S_11_C_4_1.h5' --visualize 1
#python validate_evl.py --ckpt_path 'your_model_path' --dataset_path 'your_data_path' --visualize 0