import argparse
import os
import sys 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#import torchvision.transforms as transforms

#from hemlets import HEMletsPose
from dataset_scitas import H36M
#from loss import MyLoss # replace with your own loss function
from loss import MPJPE_Loss
from HEMlets.config import config
from network import Network
import HEMlets.dataloader as dataloader
from HEMlets.model_opr import load_model
from HEMlets.getActionID import LoadSeqJsonDict
import matplotlib.pyplot as plt


def main(args):
    
    # Set up dataset and data loader
    tiny_dataset = '../data/S11/S_11_C_4_1.h5'
    train_dataset = H36M()
    train_loader = dataloader.val_loader(train_dataset, config, 0, 1)

    #train_dataset = MyDataset(args.dataset_path, transform=transforms.ToTensor()) # replace with your own dataset class
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Set up model and optimizer
    model = Network(config)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Set up loss function
    criterion = MPJPE_Loss() # replace with your own loss function

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train model
    seqList = [11]
    seqJsonDict = {}
    for seq in seqList:
        seqJsonDict[seq] =  LoadSeqJsonDict(rootPath = '../data/',subject=seq)
    for epoch in range(args.epochs):
        #for batch_idx, (data, target) in enumerate(train_loader):
        #print("HEEERE",np.shape(enumerate(train_loader)))
        count = 0
        for (idx, data) in enumerate(train_loader):
            #data, target = data.to(device), target.to(device)            

            joint2d, joint3d, path_frame = data
            print(path_frame)

            # Open the video file
            video = cv2.VideoCapture(path_frame[0])
            
            # Set the frame number to the desired frame
            video.set(cv2.CAP_PROP_POS_FRAMES, path_frame[1])
            
            # Read the frame
            ret, image = video.read()
            
            # Check if the frame was read successfully
            if not ret:
                print("Error reading frame")
                return None
            
            # Release the video object
            video.release()
        
            #print(np.shape(image))
            #print("MINMAX", torch.max(joint2d), torch.min(joint2d))
            #Show which 2d joints are important
            #print("JOINT", np.shape(joint2d), joint2d)
            joints2d_list = np.array([0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27])

            #Correspondance dictionary between joints -> layer:
            correspondance = {0:0,1:1,2:2,3:3,6:4,7:4,8:5,12:6,13:7,14:8,15:9,17:10,18:11,19:12,25:13,26:14,27:15}
            
            #https://github.com/una-dinosauria/3d-pose-baseline/issues/185
            """ H36M_NAMES = ['']*32
            H36M_NAMES[0] = 'Hip'
            H36M_NAMES[1] = 'RHip'
            H36M_NAMES[2] = 'RKnee'
            H36M_NAMES[3] = 'RFoot'
            H36M_NAMES[6] = 'LHip'
            H36M_NAMES[7] = 'LKnee'
            H36M_NAMES[8] = 'LFoot'
            H36M_NAMES[12] = 'Spine'
            H36M_NAMES[13] = 'Thorax'              p,c 13, 15      13-17-18-19     13-25-26-27     0-13    0-1-2-3     0-6-7-8
            H36M_NAMES[14] = 'Neck/Nose'
            H36M_NAMES[15] = 'Head'                         Il y en a 14, c'est juste!
            H36M_NAMES[17] = 'LShoulder'
            H36M_NAMES[18] = 'LElbow'
            H36M_NAMES[19] = 'LWrist'
            H36M_NAMES[25] = 'RShoulder'
            H36M_NAMES[26] = 'RElbow'
            H36M_NAMES[27] = 'RWrist' """
            
            #https://github.com/una-dinosauria/3d-pose-baseline/issues/185
            #plt.scatter(joint2d[0,joints2d_list,0], joint2d[0,joints2d_list,1])
            #plt.show()

            
            optimizer.zero_grad()
            output, middle_out = model(image)
            #print(np.shape(middle_out))

            #Normalization of the middle_output
            """mean = torch.mean(middle_out, axis=(2,3), keepdims=True)
            std = torch.std(middle_out, axis=(2,3), keepdims=True)
            middle_out = torch.divide(torch.subtract(middle_out, mean),std)"""
            #middle_out = torch.linalg.matrix_norm(middle_out, dim = (2,3))
            min_middle = torch.min(middle_out)
            max_middle = torch.max(middle_out)
            middle_out = torch.divide(torch.subtract(middle_out, min_middle), max_middle - min_middle)

            #Show 2djoints heatmap
            array = middle_out[0,0,:,:].detach().numpy()
            #print(np.max(array), np.min(array))
            """plt.imshow(array,cmap='hot')
            plt.colorbar()
            plt.show()"""
            

            #Create heatmap from 2D joints
            """ for i in joints2d_list:
                x = np.random.normal(joint2d[0,i,0], scale=1000, size=1000)
                y = np.random.normal(joint2d[0,i,1], scale=0.001, size=1000)
                
                heatmap, xedges, yedges = np.histogram2d(x, y, bins=64)

                # Plot the heatmap
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                if i == 0:
                    plt.clf()
                    plt.imshow(heatmap.T, origin='lower')
                    plt.colorbar()
                    plt.show()        """  

            
            from scipy.ndimage import gaussian_filter
            #print(np.shape(joints2d_list))
            heatmap = np.zeros((np.shape(joints2d_list)[0], np.shape(middle_out)[2],np.shape(middle_out)[3]))
            temp = 0
            for i in joints2d_list:
                # Define the 2D point
                point = np.array([joint2d[0,i,0], joint2d[0,i,1]])

                # Create a 2D grid of coordinates around the point
                x, y = np.meshgrid(np.arange(64), np.arange(64))
                d = np.sqrt((x-point[0])**2 + (y-point[1])**2)

                # Generate a heatmap with a Gaussian kernel
                sigma = 0.001
                #print(np.shape(heatmap[temp,:,:]))
                heatmap[temp, :, :] = (gaussian_filter(d, sigma))
                heatmap[temp, :, :] = np.subtract(np.max(heatmap[temp,:,:]), heatmap[temp,:,:])
                temp += 1
                
                #Normalize the heatmap
                #print("SHAPE",np.shape(heatmap))
                #heatmap = np.linalg.norm(heatmap)
                #print("NEWSHAPE", heatmap.shape)
                """mean = np.mean(heatmap, axis=(1,2), keepdims=True)
                std = np.std(heatmap, axis=(1,2), keepdims=True)
                heatmap = np.divide(np.subtract(heatmap, mean), std)"""
                max = np.max(heatmap, axis=(1,2), keepdims=True)
                min = np.min(heatmap, axis=(1,2), keepdims=True)
                #print(min.shape)
                #heatmap = np.divide(np.subtract(heatmap, min), np.subtract(max, min))
                heatmap = (heatmap-min)/(max-min)
                # Plot the heatmap
                """if i == 15:
                    print(point)
                    plt.imshow(heatmap[2,:,:], cmap='hot')
                    plt.colorbar()
                    plt.show()"""

            #print("HEATMAP",np.shape(heatmap))
            loss = criterion(output, joint3d, middle_out, joint2d[0,joints2d_list,:], heatmap, correspondance)#NEED TO ADD 2D JOINTS ALSO
            loss.backward()
            optimizer.step()

            print('Train Epoch: {} [(%)]\tLoss: {:.6f}'.format(
                    epoch, loss.item()))

            """ if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item())) """
            #TO BREAK
            """if count == 1:
                break
            count += 1"""

        # Save model checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_dir = os.path.join(args.checkpoint_dir, 'epoch_{}.pt'.format(epoch))
            torch.save(model.state_dict(), checkpoint_dir)

      

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train HEMletsPose model')
    parser.add_argument('--dataset_path', type=str, default='./data',
                        help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for data loader (default: 4)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status (default: 10)')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='how many epochs to wait before saving model checkpoint (default: 10)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='directory to save model checkpoints (default: ./checkpoints)')
    args = parser.parse_args()

    main(args)