import cv2
import numpy as np
import os
import sys

#Adds original git folder to path
#sys.path.append('../HEMlets/')

#Adds everything to path
sys.path.append('..')

import torch
import torch.utils.data as tData
import glob
import matplotlib.pyplot as plt
import random
import json 
import sys 
import h5py  

from HEMlets.table import *



import numpy as np
from torch.utils.data import Dataset
import cv2

systm = "laptop"  #izar,vita17,laptop
act = "Directions" #"Walking"
load_imgs = True
from_videos = True

zero_centre = True
standardize_3d = True
standardize_2d = False
Normalize = True

sample = False
Samples = np.random.randint(0,74872 if act=="Walk" else 389938, 200) #389938+135836=525774
AllCameras = False
CameraView = True 
if AllCameras:
    CameraView = True
MaxNormConctraint = False 


num_cameras = 1
input_dimension = num_cameras*2
output_dimension = 3

num_of_joints = 17 #data = np.insert(data, 0 , values= [0,0,0], axis=0 )

dataset_direcotories = {"izar":"/home/rhossein/venvs/codes/VideoPose3D/data/",
                "vita17":"/data/rh-data/h3.6/npz/", 
                "laptop": "../data/"}  #vita17 used to be: /home/rh/h3.6/dataset/npz/",
data_directory =  dataset_direcotories[systm]
path_positions_2d_VD3d = data_directory + "data_2d_h36m.npz" #"data_2d_h36m_gt.npz" 
path_positions_3d_VD3d =data_directory + "data_3d_h36m.npz"


subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
subjects = ['S7']

KeyPoints_from3d = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
# KeyPoints_from3d = list(range(32))
KeyPoints_from3d_to_delete = [4,5,9,10,11,16,20,21,22,23,24,28,29,30,31]


class H36M(Dataset):
    def __init__(self,batch_size=1, video_path = '../data/test_set/test4_cut.mp4'):
        self.batch_size = batch_size
        self.video_path = video_path

        self.video_and_frame_paths = self.read_data()       


    def __len__(self):
        return (np.shape(self.video_and_frame_paths)[0]) #number of all the frames 

    def __getitem__(self, idx):

        cap = cv2.VideoCapture(self.video_and_frame_paths[idx][0])
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.video_and_frame_paths[idx][1]) 
        res, self.frame = cap.read()
                
        # self.frame = np.divide(self.frame - img_mean, img_std)
         
        # self.frame = cv2.resize(self.frame, (256, 256))
        print(np.max(self.frame), np.min(self.frame))
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        return self.frame

    
    def read_data(self):
        print("TREAD")
            
        video_and_frame_paths = []
       
        cap = cv2.VideoCapture(self.video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        print("Total frames:", frame_count)

        for frame_num in range(10):            
            video_and_frame_paths.append( [self.video_path,frame_num*100])
                
        return video_and_frame_paths
