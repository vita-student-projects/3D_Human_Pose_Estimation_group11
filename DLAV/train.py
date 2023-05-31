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
from loss import MPJPE_Loss
from HEMlets.config import config
from network import Network
import dataloader
from HEMlets.model_opr import load_model
from dataset import H36M
from HEMlets.getActionID import LoadSeqJsonDict
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal
import h5py 
from HEMlets.table import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import imageio_ffmpeg

def plot_losses(training_loss, validation_loss, training_MPJPE, val_MPJPE):
    train_epochs = []
    train_losses = []
    val_epochs = []
    val_losses = []
    MPJPE_train_epochs=[]
    MPJPE_val_epochs=[]
    MPJPE_train=[]
    MPJPE_val=[]

    with open(training_loss, 'r') as train_file:
        for line in train_file:
            epoch, loss = line.strip().split(',')
            train_epochs.append(int(epoch))
            train_losses.append(float(loss))

    with open(validation_loss, 'r') as val_file:
        for line in val_file:
            epoch, loss = line.strip().split(',')
            val_epochs.append(int(epoch))
            val_losses.append(float(loss))
            
    with open(training_MPJPE, 'r') as val_file:
        for line in val_file:
            epoch, loss = line.strip().split(',')
            MPJPE_train_epochs.append(int(epoch))
            MPJPE_train.append(float(loss))
            
    with open(val_MPJPE, 'r') as val_file:
        for line in val_file:
            epoch, loss = line.strip().split(',')
            MPJPE_val_epochs.append(int(epoch))
            MPJPE_val.append(float(loss))

    plt.figure()
    plt.plot(train_epochs, train_losses, label='Training Loss')
    plt.plot(val_epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    name="LossPlots.png"
    plt.savefig(name)
    
    plt.figure()
    plt.plot(MPJPE_train_epochs, MPJPE_train, label='Training MPJPE')
    plt.plot(MPJPE_val_epochs, MPJPE_val, label='Validation MPJPE')
    plt.xlabel('Epoch')
    plt.ylabel('MPJPE')
    plt.title('Training and Validation MPJPE')
    plt.legend()
    name="MPJPE_Plots.png"
    plt.savefig(name)
    plt.show()

def draw_plots(joints, img, joints_gt):
    fig = plt.figure( figsize=(19.2 / 2, 10.8 / 2) )
    gs1 = gridspec.GridSpec(1, 3) # 6 rows, 10 columns
    gs1.update(left=0.08, right=0.98,top=0.95,bottom=0.08,wspace=0.05, hspace=0.1)

    font = {'family' : 'serif',  
        'color'  : 'darkred',  
        'weight' : 'normal',  
        'size'   : 10,  
            }
    
    axImg=plt.subplot(gs1[0,0])
    axImg.axis('off')
    # axImg.set_title('Input Image' )#,fontdict=font)

    axPose3d_gt=plt.subplot(gs1[0,1],projection='3d')
    axPose3d_pred=plt.subplot(gs1[0,2],projection='3d')
    img = np.transpose(img, (1,2,0))
    img = ((img * img_std) + img_mean).detach().numpy().astype(np.uint8)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axImg.imshow(img)
    axImg.axis('off')

    Draw3DSkeleton(joints, axPose3d_pred,JOINT_CONNECTIONS,'Pred_joint3d',fontdict=font,j18_color=JOINT_COLOR_INDEX,image = None)
    gt_exp = np.zeros((1, 18, 3))
    gt_exp[:, :17, :] = joints_gt
    gt_exp[:,17,:] = joints_gt[:,7,:]
    Draw3DSkeleton(gt_exp,axPose3d_gt,JOINT_CONNECTIONS,'GT_joint3d',fontdict=font,j18_color=JOINT_COLOR_INDEX,image = None)
    plt.show()
def Draw3DSkeleton(channels,ax,edge,Name=None,fontdict=None,j18_color  = None,image = None):
    edge = np.array(edge)
    I    = edge[:,0]
    J    = edge[:,1]
    LR  = np.ones((edge.shape[0]),dtype=np.int)
    colors = [(0,0,1.0),(0,1.0,0),(1.0,0,0)]
    if (torch.is_tensor(channels)):
        temp = channels.clone()
        temp = temp.detach().numpy()

        vals = np.reshape(temp, (-1, 3) )
    else:
        vals = np.reshape(channels, (-1, 3) )

    vals[:] = vals[:] - vals[0]
    # print("VLAS",vals)
    ax.cla()
    ax.view_init(azim=-136,elev=-157)
    ax.invert_yaxis()

    for i in np.arange( len(I) ):
        x,y,z = [np.array([vals[I[i],j],vals[J[i],j]]) for j in range(3)]
        ax.plot(x, -z, y, lw=2, c=colors[j18_color[i]])

    for i in range(16):
        ax.plot([vals[i,0],vals[i,0]+1],[-vals[i,2],-vals[i,2]],[vals[i,1],vals[i,1]],lw=3,c=(0.0,0.8,0.0))		

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    maxAxis = np.max(vals,axis=0)
    minAxis = np.min(vals,axis=0)
    
    max_size = 130
    ax.set_xlim3d([-max_size , max_size ])
    ax.set_ylim3d([-max_size , max_size ])
    ax.set_zlim3d([-max_size , max_size ])

    if Name is not None :
        ax.set_title(Name,fontdict=fontdict)

def from_normjoint_to_cropspace(joint3d):
    temp = np.zeros_like(joint3d)
    temp = np.copy(joint3d)
    temp[:,:,:2] = (temp[:,:,:2] + 0.5 )*256.0

    temp[:,:,2]*=128

    return temp

def create_heatmap(joint3d, middle_out_size):
    heatmap = np.zeros((np.shape(joint3d)[0], np.shape(joint3d)[1], middle_out_size, middle_out_size))
    for l in range(np.shape(joint3d)[0]):
        for k in range(np.shape(joint3d)[1]):
            # Define the 2D point
            x, y = np.array([joint3d[l,k,0], joint3d[l,k,1]])*64
            
            grid_size = 64
            grid = np.zeros((grid_size, grid_size))
            
            mean = [x, y]
            covariance = [[2, 0], [0, 2]]
            gaussian = multivariate_normal(mean=mean, cov=covariance)

            for i in range(grid_size):
                for j in range(grid_size):
                    grid[i, j] = gaussian.pdf([i, j])

            heatmap[l, k, :, :] = grid/np.max(grid)
            # plt.imshow(heatmap[l,k,:,:], cmap='hot')
            # plt.colorbar()
            # plt.show()
    return heatmap
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
        ax.invert_zaxis()
        plt.show()
def inverse_norm(skeleton, min3d, max3d):
    skeleton = skeleton.detach().numpy()
    min3d = min3d.detach().numpy()
    max3d = max3d.detach().numpy()
    rec_skeleton = np.zeros(np.shape(skeleton))
    rec_skeleton = (skeleton) * (max3d - min3d) + min3d
    return rec_skeleton

def images_crop(images, global_pos, joint3d):
    img_mean = np.array([123.675,116.280,103.530])
    img_std = np.array([58.395,57.120,57.375])
    net = cv2.dnn.readNet("../ckpt/yolov3.weights","../ckpt/yolov3.cfg")
    model_crop = cv2.dnn_DetectionModel(net)
    #Resize into a small square (320,320) to process a fast analysis
    #Scale because the dnn go from 0 to 1 and the pixel value from 0 to 255
    model_crop.setInputParams(size=(320,320), scale=1/255)

    classes = []
    with open("../ckpt/classes.txt", "r") as file_object:
        for class_name in file_object.readlines():
            #To get the good shape of inputs
            class_name = class_name.strip()
            classes.append(class_name)

    res_cropped = np.zeros(np.shape(images))
    for i in range(np.shape(images)[0]):
        image=(np.array(images[i,:,:,:].detach().numpy()))
        image= (image*img_std) + img_mean
        image = image.astype(np.uint8)

        (class_ids, score, bound_boxes) = model_crop.detect(image)
        image = image.astype(np.float)
        image = np.divide(image - img_mean, img_std)

        for class_ids, score, bound_boxes in zip(class_ids, score, bound_boxes):
            x, y, w, h = bound_boxes
            class_name=classes[int(class_ids)]
            
            if class_name=="person":

                add = 10
                image = np.copy(images[i])
                original_x = 256
                original_y = 256
                if h >= w:
                    diff = int((h-w)/2)
                    low = x - diff - add
                    low = np.clip(low, 0, original_x)
                    high = x - diff + h + add
                    high = np.clip(high, 0, original_x)

                    cropped = image[np.clip(y-add, 0, 256):np.clip(y+h+add, 0, 256),low:high,:]
                    
                else:
                    diff = int((w-h)/2)
                    low = y - diff - add
                    low = np.clip(low, 0, original_y)
                    high = y - diff + w + add
                    high = np.clip(high, 0, original_y)
                    cropped = image[low:high,np.clip(x-add, 0, 256):np.clip(x+w+add, 0, 256),:]
                res_cropped[i,:,:,:] = (cv2.resize((cropped), (256,256)))
                break
       
        # show_skeleton(global_pos)
        
    return res_cropped
def main(args):
    show = False
    trainable = True
    enable_drawing = False
    show_each = 1
    #Files to save the losses
    f = open("checkpoints/loss_train.txt", "w")
    f.close()
    f = open("checkpoints/loss_val.txt", "w")
    f.close()
    f = open("checkpoints/MPJPE_val.txt", "w")
    f.close()
    f = open("checkpoints/MPJPE_train.txt", "w")
    f.close() 

    #train_dataset = H36M(args.dataset_path, split='val')
    train_dataset = H36M(args.batch_size)

    train, validation, test = dataloader.val_loader(train_dataset, config, args.data_ratio, args.validation_ratio, args.test_ratio, args.batch_size)

    # Set up model and optimizer
    model = Network(config)

    #Load the pretrained network
    model.load_state_dict(torch.load("../ckpt/hemlets_h36m_lastest.pth", map_location = torch.device('cpu')))

    if not trainable:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Set up loss function
    criterion = MPJPE_Loss() # replace with your own loss function

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for epoch in range(args.epochs):

        loss_train = 0
        count_loss = 0
        for (idx, data) in enumerate(train):

            joint2d,joint3d, image_b, global_pos, min2d, max2d, min3d, max3d = data

            global_pos[:,:,2] = global_pos[:,:,2]/255.0 - 0.5
            global_pos[:,:,0:2] = global_pos[:,:,0:2]/256.0 - 0.5

            image = torch.from_numpy(images_crop(image_b, global_pos, joint3d)).float()

            #Show which 2d joints are important
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
            H36M_NAMES[15] = 'Head'
            H36M_NAMES[17] = 'LShoulder'
            H36M_NAMES[18] = 'LElbow'
            H36M_NAMES[19] = 'LWrist'
            H36M_NAMES[25] = 'RShoulder'
            H36M_NAMES[26] = 'RElbow'
            H36M_NAMES[27] = 'RWrist' """
            
            #https://github.com/una-dinosauria/3d-pose-baseline/issues/185

            optimizer.zero_grad()
            image = np.transpose(image, (0,3,1,2))
            image_b = np.transpose(image_b, (0,3,1,2))
            image_b = (image_b).float()

            output, middle_out = model(image)
            output[:,:,:2] = (output[:,:,:2] + 0.5)*256.0
            output[:,:,2] *= 128

            global_pos = global_pos[:,:,[0,2,1]]
            global_pos[:,:,1] = -global_pos[:,:,1]
            gt_joints = from_normjoint_to_cropspace(global_pos) * 128
            if enable_drawing:
                if (epoch-1) % show_each == 0:
                    for i in range(np.shape(image)[0]):
                        draw_plots(output[i:i+1,:,:], image[i], gt_joints[i:i+1,:,:])

            if not trainable:
                output = from_normjoint_to_cropspace(output)
                gt_joints = from_normjoint_to_cropspace(global_pos)
                draw_plots(output, image[0], gt_joints)
           
            min_middle = torch.min(middle_out)
            max_middle = torch.max(middle_out)
            middle_out = torch.divide(torch.subtract(middle_out, min_middle), max_middle - min_middle)

            #Show 2djoints heatmap
            array = middle_out[0,0,:,:].detach().numpy()
            if show:
                plt.imshow(array,cmap='hot')
                plt.colorbar()
                plt.show()
            

            heatmap = create_heatmap(joint3d, np.shape(middle_out)[2])
            if show and False:
                plt.imshow(heatmap[0,0,:,:], cmap='hot')
                plt.colorbar()
                plt.show()

            loss, MPJPE_train = criterion(output, gt_joints, middle_out, joint2d)
            MPJPE_train = MPJPE_train * 10
            # loss = criterion(output, joint3d, middle_out, joint2d)
            if trainable:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('Train Epoch: {} [(%)]\tLoss: {:.6f}'.format(
                    epoch, loss.item()))
            #TO BREAK
            """if count == 1:
                break
            count += 1"""
            count_loss += 1
            loss_train += loss.item()
        for (idx, data) in enumerate(validation):
            # joint2d,joint3d, image_bv = data
            joint2d,joint3d, image_bv, global_pos_v, min2d, max2d, min3d, max3d = data

            image = torch.from_numpy(images_crop(image_bv, global_pos_v, joint3d)).float()
            #print(np.shape(image))
            image = np.transpose(image, (0,3,1,2))
            val_output, middle_out = model(image)
            min_middle = torch.min(middle_out)
            max_middle = torch.max(middle_out)
            middle_out = torch.divide(torch.subtract(middle_out, min_middle), max_middle - min_middle)

            heatmap = create_heatmap(joint3d, np.shape(middle_out)[2])

            # val_output = from_normjoint_to_cropspace(val_output)
            val_output[:,:,:2] = (val_output[:,:,:2] + 0.5)*256.0
            val_output[:,:,2] *= 128

            global_pos_v[:,:,2] = global_pos_v[:,:,2]/255.0 - 0.5
            global_pos_v[:,:,0:2] = global_pos_v[:,:,0:2]/256.0 - 0.5

            global_pos_v = global_pos_v[:,:,[0,2,1]]
            global_pos_v[:,:,1] = -global_pos_v[:,:,1]
            gt_joints = from_normjoint_to_cropspace(global_pos_v) * 128
            
            if enable_drawing:
                if (epoch-1) % show_each == 0:
                    for i in range(np.shape(image)[0]):
                        draw_plots(val_output[i:i+1,:,:], image[i], gt_joints[i:i+1,:,:])

            loss_val, MPJPE_val = criterion(val_output, gt_joints, middle_out, joint2d)
            MPJPE_val = MPJPE_val * 100
            # loss_val = criterion(val_output, joint3d, middle_out, joint2d)
            print("Loss val",loss_val.item())
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

            f = open("checkpoints/MPJPE_train.txt", "a")
            f.write(repr(epoch) + ", " + repr(MPJPE_train) + "\n")
            f.close()
            
            f = open("checkpoints/MPJPE_val.txt", "a")
            f.write(repr(epoch) + ", " + repr(MPJPE_val) + "\n")
            f.close()

        if (epoch+1  == args.epochs):
            print("DONE")
            training_filename = 'checkpoints/loss_train.txt'  # Replace with your training loss file name
            validation_filename = 'checkpoints/loss_val.txt'  # Replace with your validation loss file name
            val_MPJPE_file = 'checkpoints/MPJPE_val.txt'  # Replace with your validation loss file name
            train_MPJPE_file = 'checkpoints/MPJPE_train.txt'  # Replace with your validation loss file name
            plot_losses(training_filename, validation_filename,train_MPJPE_file,val_MPJPE_file)

            


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
