#!/bin/bash
#conda activate dlav

cd RealTime
python validate_evl.py --ckpt_path '../ckpt/hemlets_h36m_lastest.pth' --dataset_path '../data/S11/S_11_C_4_1.h5' --visualize 1
