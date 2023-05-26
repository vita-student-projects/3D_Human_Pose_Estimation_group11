import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#import torchvision.transforms as transforms

#from hemlets import HEMletsPose
from dataset import H36M
#from loss import MyLoss # replace with your own loss function
from loss_lifting import MPJPE_Loss
from HEMlets.config import config
from network_lifting import Network
import dataloader
from HEMlets.model_opr import load_model
from dataset import H36M
from HEMlets.getActionID import LoadSeqJsonDict
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal

def show_skeleton(joints):
    for i in range(np.shape(joints)[0]):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot of the data points

        # Set labels for the axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        keypoints = joints[i]
        xdata = keypoints[:,0]
        ydata = keypoints[:,1]
        zdata = keypoints[:,2]
        ax.scatter(xdata,ydata,zdata,"b",label="expectations")
        sk_points = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]
        for j in range(17):
            ax.plot(xdata[sk_points[j]], ydata[sk_points[j]], zdata[sk_points[j]] , "b" )
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.show()
def inverse_norm(skeleton, min3d, max3d):
    skeleton = skeleton.detach().numpy()
    min3d = min3d.detach().numpy()
    max3d = max3d.detach().numpy()
    rec_skeleton = np.zeros(np.shape(skeleton))
    # for i in range(np.shape(skeleton)[0]):
    #     rec_skeleton[i] = (np.add(skeleton[i], min3d[i]))*(max3d[i] - min3d[i])
    rec_skeleton = (skeleton) * (max3d - min3d) + min3d
    return rec_skeleton
    rec_skeleton[:,:,2] = (skeleton[:,:,2] + 0.5) * 255.0
    rec_skeleton[:,:,0:2] = (skeleton[:,:,0:2] + 0.5) * 256.0
    return rec_skeleton


def main(args):
    show = False
    #Files to save the losses
    f = open("checkpoints/loss_train.txt", "w")
    f.close()
    f = open("checkpoints/loss_val.txt", "w")
    f.close()

    # Set up dataset and data loader
    tiny_dataset = '../data/S11/S_11_C_4_1.h5'
     
    #train_dataset = H36M(args.dataset_path, split='val')
    train_dataset = H36M(args.batch_size)

    train, validation, test = dataloader.val_loader(train_dataset, config, args.data_ratio, args.validation_ratio, args.test_ratio, args.batch_size)

    #train_dataset = MyDataset(args.dataset_path, transform=transforms.ToTensor()) # replace with your own dataset class
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Set up model and optimizer
    model = Network(config)
    #model.load_state_dict(torch.load("../ckpt/hemlets_h36m_lastest.pth", map_location = torch.device('cpu')))
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
    temp = 0
    for epoch in range(args.epochs):

        loss_train = 0
        count_loss = 0
        for (idx, data) in enumerate(train):

            joint2d,joint3d, image_b, global_pos, min2d, max2d, min3d, max3d = data

            optimizer.zero_grad()
            output = model(joint2d)
            # Extracting x, y, and z coordinates
            reconstructed_skeleton = inverse_norm(output, min3d, max3d)
            reconstructed_global_pos = inverse_norm(global_pos, min3d, max3d)
            if temp == 5:
                temp = 0
                show_skeleton(global_pos[:,:17,:])
                show_skeleton(reconstructed_skeleton[:,:17,:])
                show_skeleton(output[:,:17,:].detach().numpy())
            temp += 1
            loss = criterion(output, global_pos, joint2d)
            print("LOSS",loss.item())
            optimizer.zero_grad()
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
            count_loss += 1
            loss_train += loss.item()
        for (idx, data) in enumerate(validation):
            # joint2d,joint3d, image_bv = data
            joint2d,joint3d, image_bv, global_pos_v, min2d, max2d, min3d, max3d = data

            #print(np.shape(image))
            val_output = model(joint2d)

            loss_val = criterion(val_output, global_pos, joint2d)
            print("LOSSSSSSSSVAAAL",loss_val.item())
        # Getting all memory using os.popen()
        total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        
        # Memory usage
        print("RAM memory percent used:", round((used_memory/total_memory) * 100, 2), total_memory)
        # Save model checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_dir = os.path.join(args.checkpoint_dir, 'epoch_{}.pt'.format(epoch))
            torch.save(model.state_dict(), checkpoint_dir)

            loss_train = loss_train/count_loss
            print("Average loss on epoch",loss_train)
            f = open("checkpoints/loss_train.txt", "a")
            f.write(repr(epoch) + ", " + repr(loss_train) + "\n")
            f.close()

            f = open("checkpoints/loss_val.txt", "a")
            f.write(repr(epoch) + ", " + repr(loss_val.item()) + "\n")
            f.close()

      

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
    parser.add_argument('--data_ratio', type=float, default=0.1,
                        help='Percentage of data taken')
    parser.add_argument('--test_ratio', type=float, default=0.3,
                        help='percentage of data for testing')
    parser.add_argument('--validation_ratio', type=float, default=0.1,
                        help='Percentage of data for validation')
    args = parser.parse_args()

    main(args)
