import cv2
import numpy as np
import os
import sys

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
from utils import camera_parameters, qv_mult
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
    def __init__(self, num_cams = 1, subjectp=subjects , action=act, transform=None, target_transform=None, is_train = True, batch_size=1):
        self.batch_size = batch_size
        self.cam_ids = [".54138969", ".55011271", ".58860488",  ".60457274", ]
        self.cam_ids = [".60457274"]

        self.dataset2d, self.dataset3d, self.video_and_frame_paths, self.global_pos = self.read_data(subjects= subjectp,action=action,is_train = is_train)
        self.dataset2d, self.min2d, self.max2d = self.process_data(self.dataset2d,  sample = False if len(subjectp)==2 else sample, is_train = is_train, standardize=standardize_2d, z_c = False)
        self.dataset3d, self.min3d, self.max3d = self.process_data(self.dataset3d,  sample = False if len(subjectp)==2 else sample, is_train = is_train, standardize=standardize_3d, z_c = True)

        self.transform = transform
        self.target_transform = target_transform
        self.num_cams = num_cams
        self.is_train = is_train

        self.frame =  np.zeros((1000,1002,3))
        
        


    def __len__(self):
        return (np.shape(self.video_and_frame_paths)[0]) #number of all the frames 

    def __getitem__(self, idx):
        img_mean = np.array([123.675,116.280,103.530])
        img_std = np.array([58.395,57.120,57.375])

        if load_imgs:
            if from_videos:
                cap = cv2.VideoCapture(self.video_and_frame_paths[idx][0])
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.video_and_frame_paths[idx][1]) 
                res, self.frame = cap.read()
                self.frame = np.divide(self.frame - img_mean, img_std)
            else :
                self.frame = cv2.imread(self.video_and_frame_paths[idx][0])
 
                self.frame = np.divide(self.frame - img_mean, img_std)

        keypoints_2d = self.dataset2d[idx].reshape(-1 ,2)
        #resising the image for Resnet
        self.frame = cv2.resize(self.frame, (256, 256))

        return keypoints_2d, self.dataset3d[idx], self.frame.astype(np.float), self.global_pos[idx], self.min2d[idx], self.max2d[idx], self.min3d[idx], self.max3d[idx] #cam 0 

        
    def process_data(self, dataset , sample=sample, is_train = True, standardize = False, z_c = zero_centre) :
        n_frames, n_joints, dim = dataset.shape
        if dim == 3:
            max3d = np.zeros((np.shape(dataset)[0], np.shape(dataset)[2]))
            min3d = np.zeros((np.shape(dataset)[0], np.shape(dataset)[2]))
            
            # for i in range(n_frames):
            max3d = np.max(dataset, axis = (1))
            min3d = np.min(dataset, axis = (1))
            max3d_exp = np.expand_dims(max3d, axis=1)
            min3d_exp = np.expand_dims(min3d, axis=1)
            dataset = (dataset- min3d_exp)/(max3d_exp - min3d_exp)
            return dataset, min3d_exp, max3d_exp
        else:
            max2d = np.zeros((np.shape(dataset)[0], np.shape(dataset)[2]))
            min2d = np.zeros((np.shape(dataset)[0], np.shape(dataset)[2]))
  
            max2d = np.max(dataset, axis = (1))
            min2d = np.min(dataset, axis = (1))
            max2d_exp = np.expand_dims(max2d, axis=1)
            min2d_exp = np.expand_dims(min2d, axis=1)
            dataset = (dataset- min2d_exp)/(max2d_exp - min2d_exp)

            return dataset, min2d_exp, max2d_exp
        
    
    def read_data(self,subjects = subjects, action = "", is_train=True):
        print("TREAD")
        data_file_3d = np.load(path_positions_3d_VD3d, allow_pickle=True)
        data_file_2d = np.load(path_positions_2d_VD3d, allow_pickle=True)

        data_file_3d = data_file_3d['positions_3d'].item()
        data_file_2d = data_file_2d['positions_2d'].item()

        n_frame = 0 
        number_image_per_video = 10
        for s in subjects:
            for a in [action]:
            # for a in data_file_3d[s].keys():
                if (action in a ) :
                    n_frame += number_image_per_video
                    # n_frame += len(data_file_3d[s][a])  
            
        all_in_one_dataset_3d = np.zeros((4*n_frame if AllCameras else n_frame, 17 ,3),  dtype=np.float32)
        all_in_one_dataset_2d = np.zeros((4*n_frame if AllCameras else n_frame, 17 ,2),  dtype=np.float32)
        global_pose = np.zeros((4*n_frame if AllCameras else n_frame, 17 ,3),  dtype=np.float32)
        video_and_frame_paths = []
        i = 0
        for s in subjects:
            for a in [action]:
            # for a in data_file_3d[s].keys():
                if action in a :
                    print(np.shape(data_file_3d[s][a][0]))
                    print(s,a,number_image_per_video)
                    # print(s,a,len(data_file_3d[s][a]))
                    for frame_num in range(number_image_per_video):#range(len(data_file_3d[s][a])):
                        #print("FRAME",frame_num)
                        frame_number = 100
                        global_pose_one_img = data_file_3d[s][a][frame_num*frame_number]  
                        global_pose[frame_num] = global_pose_one_img[KeyPoints_from3d ,:] #only keeping the 16 or 17 keypoints we want

                        for c in range(1+3*int(AllCameras)) :

                            tmp = global_pose[frame_num].copy()

                            if CameraView:
                                for j in range(len(tmp)): 
                                    tmp[j] = tmp[j] - np.divide(np.array(camera_parameters[s][c]['translation']),1000)
                                    tmp[j] = qv_mult(np.array(camera_parameters[s][c]['orientation']),tmp[j])
                                        
                            all_in_one_dataset_3d[i] = tmp

                            tmp2 = data_file_2d[s][a+self.cam_ids[c]][frame_num*frame_number]
                            
                            all_in_one_dataset_2d[i] = tmp2[ KeyPoints_from3d ,:] #only keeping the 16 or 17 keypoints we want

                            if load_imgs:
                                #print("LOAD")
                                if from_videos:
                                    video_and_frame_paths.append( ["../data/"+a+self.cam_ids[c]+".mp4",frame_num*frame_number])
                                else:
                                    if systm == "laptop":
                                        video_and_frame_paths.append( ["../data/"+a+self.cam_ids[c]+".mp4",frame_num*frame_number])
                                        # video_and_frame_paths.append( ["../data/"+a+self.cam_ids[c]+".mp4/"+str(frame_num+1).zfill(4)+".jpg",frame_num])
                                        #print(video_and_frame_paths)
                                    else:
                                        video_and_frame_paths.append( ["/data/rh-data/h3.6/videos/"+s+"/outputVideos/"+a+self.cam_ids[c]+".mp4/"+str(frame_num*frame_number+1).zfill(4)+".jpg",frame_num*frame_number])

                            i = i + 1 

        all_in_one_dataset_2d[:,:,1] = 1 - all_in_one_dataset_2d[:,:,1]
        return all_in_one_dataset_2d, all_in_one_dataset_3d , video_and_frame_paths, global_pose