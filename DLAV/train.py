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
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.show()
def inverse_norm(skeleton, min3d, max3d):
    skeleton = skeleton.detach().numpy()
    print(type(max3d), type(skeleton))
    min3d = min3d.detach().numpy()
    max3d = max3d.detach().numpy()
    rec_skeleton = np.zeros(np.shape(skeleton))
    print(np.shape(skeleton),np.shape(min3d))
    # for i in range(np.shape(skeleton)[0]):
    #     rec_skeleton[i] = (np.add(skeleton[i], min3d[i]))*(max3d[i] - min3d[i])
    rec_skeleton = (skeleton) * (max3d - min3d) + min3d
    return rec_skeleton
    rec_skeleton[:,:,2] = (skeleton[:,:,2] + 0.5) * 255.0
    rec_skeleton[:,:,0:2] = (skeleton[:,:,0:2] + 0.5) * 256.0
    return rec_skeleton


def images_crop(images, global_pos, joint3d):
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

    res_cropped = np.zeros(np.shape(np.transpose(images, (0,3,2,1))))
    for i in range(np.shape(images)[0]):
        image=(np.array(np.transpose(images[i,:,:,:].detach().numpy())))
        image=image*255
        image = image.astype(np.uint8)

        (class_ids, score, bound_boxes) = model_crop.detect(np.transpose(image)) 
        # plt.imshow(np.transpose(images[i,:,:,:]))
        # plt.show()
        image = image.astype(np.float)
        image = image / 255

        for class_ids, score, bound_boxes in zip(class_ids, score, bound_boxes):
            x, y, w, h = bound_boxes
            #print(x, y, h, w)
            class_name=classes[int(class_ids)]
            
            if class_name=="person":
                #cv2.putText(image, str(class_name)+str(score), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 3, (200, 0, 50), 2)
                #cv2.rectangle(image, (x,y), (x+w,y+h), (200, 0, 50), 3)
                #cv2.imshow("Frame", image)
                #cv2.waitKey(0)
                #print(np.shape(image))
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
                res_cropped[i,:,:,:] = np.transpose(cv2.resize((cropped), (256,256)))
                break
        

        x = global_pos[i,:, 2]
        y = global_pos[i,:, 0]
        z = -global_pos[i,:, 1]
        print(np.shape(global_pos))
       
        # show_skeleton(global_pos)
        
    return res_cropped
def main(args):
    show = False
    trainable = False
    #Files to save the losses
    f = open("checkpoints/loss_train.txt", "w")
    f.close()
    f = open("checkpoints/loss_val.txt", "w")
    f.close()

    # Set up dataset and data loader
    tiny_dataset = '../data/S11/S_11_C_4_1.h5'
    img_mean = np.array([123.675,116.280,103.530])
    img_std = np.array([58.395,57.120,57.375])
    file = h5py.File(tiny_dataset, 'r')
    
    img = file['images']
    img = np.divide((img - img_mean), img_std)
    img = np.transpose(img, (0,3,1,2)) #WORKS
    # img = np.transpose(img, (0,2,1,3))
    print("GOOOD_SHAPE",np.shape(img))
    # np.transpose(image,(2,0,1))
    
    print("IMASDhbfJSbf", np.shape(img))
     
    #train_dataset = H36M(args.dataset_path, split='val')
    train_dataset = H36M(args.batch_size)

    train, validation, test = dataloader.val_loader(train_dataset, config, args.data_ratio, args.validation_ratio, args.test_ratio, args.batch_size)

    #train_dataset = MyDataset(args.dataset_path, transform=transforms.ToTensor()) # replace with your own dataset class
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Set up model and optimizer
    model = Network(config)

    #Load the pretrained network
    model.load_state_dict(torch.load("../ckpt/hemlets_h36m_lastest.pth", map_location = torch.device('cpu')))

    if not trainable:
        for param in model.parameters():
            param.requires_grad = False

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

        loss_train = 0
        count_loss = 0
        for (idx, data) in enumerate(train):

            joint2d,joint3d, image_b, global_pos, min2d, max2d, min3d, max3d = data

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
            H36M_NAMES[15] = 'Head'                         Il y en a 14, c'est juste!
            H36M_NAMES[17] = 'LShoulder'
            H36M_NAMES[18] = 'LElbow'
            H36M_NAMES[19] = 'LWrist'
            H36M_NAMES[25] = 'RShoulder'
            H36M_NAMES[26] = 'RElbow'
            H36M_NAMES[27] = 'RWrist' """
            
            #https://github.com/una-dinosauria/3d-pose-baseline/issues/185
            # plt.scatter(joint2d[0, :, 0], joint2d[0,:,1])
            # plt.show()

            optimizer.zero_grad()
            image = np.transpose(image, (0,1,3,2))
            image_b = np.transpose(image_b, (0,3,1,2))
            image_b = (image_b).float()
            print("OURs / crop / REAL", np.shape(image), np.shape(image_b), np.shape(torch.tensor(img[0:1])))

            output, middle_out = model(image_b)
            show_skeleton(output[:,:17,:].detach().numpy())
            plt.imshow(np.transpose(img[0], (1,2,0)))
            plt.show()
            # print(np.max(image))
            print("ICI",np.max(img), np.min(img))
            print("ICI2", torch.max(image), torch.min(image))
            print("ICI3", torch.max(image_b), torch.min(image_b))

            plt.imshow(np.transpose(image_b[0], (1,2,0)))
            plt.show()
            output, middle_out = model(torch.tensor(img[0:1]).float())
            # Extracting x, y, and z coordinates
            reconstructed_skeleton = inverse_norm(output, min3d, max3d)
            reconstructed_global_pos = inverse_norm(global_pos, min3d, max3d)
            show_skeleton(global_pos[:,:17,:])
            show_skeleton(output[:,:17,:].detach().numpy())
            show_skeleton(reconstructed_global_pos[:,:17,:])
            show_skeleton(reconstructed_skeleton[:,:17,:])

            min_middle = torch.min(middle_out)
            max_middle = torch.max(middle_out)
            middle_out = torch.divide(torch.subtract(middle_out, min_middle), max_middle - min_middle)

            #Show 2djoints heatmap
            array = middle_out[0,0,:,:].detach().numpy()
            #print(np.max(array), np.min(array))
            if show:
                plt.imshow(array,cmap='hot')
                plt.colorbar()
                plt.show()
            

            #print(np.shape(joints2d_list))
            heatmap = create_heatmap(joint3d, np.shape(middle_out)[2])
            if show and False:
                plt.imshow(heatmap[0,0,:,:], cmap='hot')
                plt.colorbar()
                plt.show()

            loss = criterion(output, global_pos, middle_out, joint2d)
            # loss = criterion(output, joint3d, middle_out, joint2d)
            if trainable:
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

            image = torch.from_numpy(images_crop(image_bv, global_pos_v, joint3d)).float()
            #print(np.shape(image))
            val_output, middle_out = model(image)
            
            min_middle = torch.min(middle_out)
            max_middle = torch.max(middle_out)
            middle_out = torch.divide(torch.subtract(middle_out, min_middle), max_middle - min_middle)

            heatmap = create_heatmap(joint3d, np.shape(middle_out)[2])


            loss_val = criterion(val_output, global_pos, middle_out, joint2d)
            # loss_val = criterion(val_output, joint3d, middle_out, joint2d)
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
