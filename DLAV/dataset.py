# import cv2
# import numpy as np
# import os
# import sys

# #Adds original git folder to path
# #sys.path.append('../HEMlets/')

# #Adds everything to path
# sys.path.append('..')

# import torch
# import torch.utils.data as tData
# import glob
# import matplotlib.pyplot as plt
# import random
# import json 
# import sys 
# import h5py  

# from HEMlets.table import *

# class H36M(tData.Dataset):
#     def __init__(self,h5_path,video_id=1,subject=11,patch_width=256,patch_height=256,split = 'train'):

#         super(H36M,self).__init__()

#         self.subject = subject

#         self.h5_path = h5_path

#         self.video_id = video_id


#         with h5py.File(self.h5_path,'r') as db:
#             self.len_data = db['images'].shape[0]

#         print(self.len_data)

#     def __len__(self):
#         return self.len_data 

#     def imgNormalize(self,img,flag = True):
#         if flag:
#             img = img[:,:,[2,1,0]]
#         return np.divide(img - img_mean, img_std)

#     def __getitem__(self, index):
#         # index = 555
#         h5_path =  self.h5_path
        
#         idx = index % self.__len__()
#         with h5py.File(h5_path,'r') as db:
#             joints = db['pos3D'][idx]
#             image = db['images'][idx]
#             #print("JOINTS", joints)
#             joints = joints/1000 #SEEMS CORRECT
#             # joints[:,2] = joints[:,2] / 255.0 - 0.5
#             # joints[:,0:2] = joints[:,0:2] / 256.0 - 0.5
#             joint3d = torch.from_numpy(joints).float()
#             image = (image).astype(float)
#             #print(type(image[0,0,0]))
#             #print(np.max(image))
#             #image = self.imgNormalize(image)
#             image = np.transpose(image,(2,1,0))
#             image = torch.from_numpy(image).float()

#             return image, joint3d
        
#         # index = 555
#         h5_path =  self.h5_path
        
#         idx = index % self.__len__()
#         with h5py.File(h5_path,'r') as db:
#             #print(db.keys())
#             joints3dCam = db['joints3d_cam'][idx] # load the camera space 3d joint (32,3)
#             joint3d_j18 = np.zeros((18,3),dtype=float)
#             joint3d_j18[0:17,:] = joints3dCam[H36M_TO_J18,:] 
#             joint3d_j18[17] = (joint3d_j18[11] + joint3d_j18[14]) * 0.5 


#             img = db['images'][idx]
#             joints = db['joints'][idx]
#             print(np.shape(img), np.shape(joints))
#             trans = db['trans'][idx]
#             camid = db['camid'][idx]

#             cam_id = np.zeros((4,),dtype = int)
#             cam_id[:3] = camid
#             cam_id[3] = self.video_id #int( ((h5_path.split('/')[-1]).split('.')[0]).split('_')[4])


            
#             joints[:,2] = joints[:,2] / 255.0 - 0.5
#             joints[:,0:2] = joints[:,0:2] / 256.0 - 0.5

#             image = self.imgNormalize(img)

#             joint3d = torch.from_numpy(joints).float()
#             joint3d_j18 = torch.from_numpy(joint3d_j18).float()
#             image = np.transpose(image,(2,0,1))

#             image_filp = image[:,:,::-1].copy()
#             image_filp = torch.from_numpy(image_filp).float()
           
#             image = torch.from_numpy(image).float()

#             #PAS SUR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#             joint2d = db['joints2d_full'][idx]/1000*64
            
#             return image,image_filp,trans,cam_id, joint2d,joint3d,joint3d_j18,np.zeros((3),dtype=float),'subject_{}'.format(self.subject)
# if __name__ == '__main__':
#     d = H36M(split = 'val')
#     for _ in range(100):
#         d[12000]
#         input('check')


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
        print(np.shape(self.dataset3d))
        self.dataset2d = self.process_data(self.dataset2d,  sample = False if len(subjectp)==2 else sample, is_train = is_train, standardize=standardize_2d, z_c = False)
        self.dataset3d = self.process_data(self.dataset3d,  sample = False if len(subjectp)==2 else sample, is_train = is_train, standardize=standardize_3d, z_c = True)

        self.transform = transform
        self.target_transform = target_transform
        self.num_cams = num_cams
        self.is_train = is_train

        self.frame =  np.zeros((1000,1002,3))
        
        


    def __len__(self):
        return (np.shape(self.video_and_frame_paths)[0]) #number of all the frames 

    def __getitem__(self, idx):
        if load_imgs:
            if from_videos:
                print(self.video_and_frame_paths)
                cap = cv2.VideoCapture(self.video_and_frame_paths[idx][0])
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.video_and_frame_paths[idx][1]) 
                res, self.frame = cap.read()
                #print(np.shape(self.frame), self.frame)
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            else :
                print(self.video_and_frame_paths)
                self.frame = cv2.imread(self.video_and_frame_paths[idx][0])
                #print(np.shape(self.frame), self.frame)
                # print("***",self.video_and_frame_paths[idx][0], self.frame ,"***")
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            
        keypoints_2d = self.dataset2d[idx].reshape(-1 ,2)

        #resising the image for Resnet
        self.frame = cv2.resize(self.frame, (256, 256))
        self.frame = self.frame/256.0

        return keypoints_2d, self.dataset3d[idx], self.frame, self.global_pos #cam 0 

        
    def process_data(self, dataset , sample=sample, is_train = True, standardize = False, z_c = zero_centre) :

        n_frames, n_joints, dim = dataset.shape

        if z_c:
            for i in range(n_frames):
                dataset[i,1:] = dataset[i,1:] - dataset[i,0]

        if is_train :
            data_sum = np.sum(dataset, axis=0)
            data_mean = np.divide(data_sum, n_frames)


            diff_sq2_sum =np.zeros((n_joints,dim))
            for i in range(n_frames):
                diff_sq2_sum += np.power( dataset[i]-data_mean ,2)
            data_std = np.divide(diff_sq2_sum, n_frames)
            data_std = np.sqrt(data_std)

            
            if dim == 2:
                with open("mean_train_2d.npy","wb") as f:
                    np.save(f, data_mean)
                with open("std_train_2d.npy","wb") as f:
                    np.save(f, data_std)  


            elif dim == 3:
                with open("mean_train_3d.npy","wb") as f:
                    np.save(f, data_mean)  
                with open("std_train_3d.npy","wb") as f:
                    np.save(f, data_std)  
                    
                with open("max_train_3d.npy","wb") as f:
                    np.save(f, np.max(dataset, axis=0))  
                with open("min_train_3d.npy","wb") as f:
                    np.save(f, np.min(dataset, axis=0)) 


        if dim == 2:
            with open("mean_train_2d.npy","rb") as f:
                mean_train_2d = np.load(f)
            with open("std_train_2d.npy","rb") as f:
                std_train_2d = np.load(f)  
        elif dim == 3:
            with open("mean_train_3d.npy","rb") as f:
                mean_train_3d =np.load(f)  
            with open("std_train_3d.npy","rb") as f:
                std_train_3d = np.load(f)  
                
            with open("max_train_3d.npy","rb") as f:
                max_train_3d =np.load(f)  
            with open("min_train_3d.npy","rb") as f:
                min_train_3d = np.load(f)  


        if standardize :
            if dim == 2 :
                for i in range(n_frames):
                    if Normalize:
                        # max_dataset, min_dataset = np.max(dataset, axis=0), np.min(dataset, axis=0)
                        # print(max_dataset, min_dataset)
                        # dataset[i] = np.divide(2*dataset[i], (max_dataset-min_dataset))
                        # dataset[i] = dataset[i] - np.divide(min_dataset, (max_dataset-min_dataset))
                        dataset[i] = 2*dataset[i] -1 

                    else:
                        dataset[i] = np.divide(dataset[i] - mean_train_2d, std_train_2d)
            elif dim == 3:
                for i in range(n_frames):
                    if Normalize:
                        # max_dataset, min_dataset = np.max(dataset, axis=0), np.min(dataset, axis=0)
                        dataset[i] = np.divide(dataset[i]- min_train_3d, (max_train_3d-min_train_3d))
                    else:
                        dataset[i] = np.divide(dataset[i] - mean_train_3d, std_train_3d)


        if num_of_joints == 16: #Should through an error if num of joints is 16 but zero centre is false    
            dataset = dataset[:, 1:, :].copy()
        elif z_c :
            dataset [:,:1,:] *= 0


        if dim == 2 and sample :
            dataset = dataset.reshape((int(n_frames/4),4, num_of_joints,2))

        dataset = dataset[Samples] if sample else dataset

        if dim == 2 and sample :
            dataset = dataset.reshape(-1, num_of_joints,2)  

        return dataset
    
    
    def read_data(self,subjects = subjects, action = "", is_train=True):
        print("TREAD")
        data_file_3d = np.load(path_positions_3d_VD3d, allow_pickle=True)
        data_file_2d = np.load(path_positions_2d_VD3d, allow_pickle=True)

        data_file_3d = data_file_3d['positions_3d'].item()
        data_file_2d = data_file_2d['positions_2d'].item()

        n_frame = 0 
        for s in subjects:
            for a in [action]:
            # for a in data_file_3d[s].keys():
                if (action in a ) :
                    n_frame += len(data_file_3d[s][a])  
            

        all_in_one_dataset_3d = np.zeros((4*n_frame if AllCameras else n_frame, 17 ,3),  dtype=np.float32)
        all_in_one_dataset_2d = np.zeros((4*n_frame if AllCameras else n_frame, 17 ,2),  dtype=np.float32)
        video_and_frame_paths = []
        i = 0
        for s in subjects:
            for a in [action]:
            # for a in data_file_3d[s].keys():
                if action in a :
                    print(np.shape(data_file_3d[s][a][0]))
                    print(s,a,len(data_file_3d[s][a]))
                    for frame_num in range(10):#range(len(data_file_3d[s][a])):
                        #print("FRAME",frame_num)
                        frame_number = 100
                        global_pose = data_file_3d[s][a][frame_num*frame_number]  
                        global_pose = global_pose[ KeyPoints_from3d ,:] #only keeping the 16 or 17 keypoints we want

                        for c in range(1+3*int(AllCameras)) :

                            tmp = global_pose.copy()

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

        
        return all_in_one_dataset_2d, all_in_one_dataset_3d , video_and_frame_paths, global_pose