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
        # Creating the 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot of the data points

        # Set labels for the axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax1.scatter(x1, y1, z1, c='b', marker='o')
        # ax.scatter(x, y, z, c='b', marker='o')
        # plt.show()
        keypoints = global_pos[i]
        xdata = keypoints[:,0]
        ydata = keypoints[:,1]
        zdata = keypoints[:,2]
        ax.scatter(xdata,ydata,zdata,"b",label="expectations")
        sk_points = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]
        print("TA",np.shape(sk_points), np.shape(xdata), np.shape(global_pos), np.shape(keypoints))
        for j in range(17):
            ax.plot(xdata[sk_points[j]], ydata[sk_points[j]], zdata[sk_points[j]] , "b" )
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        keypoints = joint3d[i]
        xdata = keypoints[:,0]
        ydata = keypoints[:,1]
        zdata = keypoints[:,2]
        ax1.scatter(xdata,ydata,zdata,"b",label="expectations")
        sk_points = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]
        print("TA",np.shape(sk_points), np.shape(xdata), np.shape(global_pos), np.shape(keypoints))
        for j in range(17):
            ax1.plot(xdata[sk_points[j]], ydata[sk_points[j]], zdata[sk_points[j]] , "b" )

        #plt.imshow((np.transpose(image[0])))
        #plt.show() here
        #plt.imshow(np.transpose(res_cropped[i]))
        #plt.show()
        # plt.show()
    return res_cropped
def main(args):
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
        loss_train = 0
        count_loss = 0
        for (idx, data) in enumerate(train):

            #data, target = data.to(device), target.to(device)            
            #image_b,joint3d = data
            #joint2d,joint3d, image_b = data
            joint2d,joint3d, image_b, global_pos = data
            print(np.shape(joint3d), np.shape(global_pos))
            image = torch.from_numpy(images_crop(image_b, global_pos, joint3d)).float()
            #image = torch.transpose(image,1,3).float()


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
            #plt.scatter(joint2d[0,joints2d_list,0], joint2d[0,joints2d_list,1])
            #plt.show()

            optimizer.zero_grad()
            output, middle_out = model(image)
            # Extracting x, y, and z coordinates


            # Show the plot
            if epoch == 90:
                x1 = output[0,:, 2].detach().numpy()
                y1 = output[0,:, 0].detach().numpy()
                z1 = -output[0,:, 1].detach().numpy()

                # Creating the 3D plot
                fig = plt.figure()
                ax1 = fig.add_subplot(111, projection='3d')

                # Scatter plot of the data points
                

                # Set labels for the axes
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
                ax1.set_zlabel('Z')

                # Show the plot
                
                x = joint3d[0,:, 2]
                y = joint3d[0,:, 0]
                z = -joint3d[0,:, 1]

                # Creating the 3D plot
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                # Scatter plot of the data points
                

                # Set labels for the axes
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax1.scatter(x1, y1, z1, c='b', marker='o')
                ax.scatter(x, y, z, c='b', marker='o')
                #plt.imshow((np.transpose(image[0])))
                #plt.show()
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
            heatmap = np.zeros((np.shape(joint3d)[0], np.shape(joints2d_list)[0], np.shape(middle_out)[2],np.shape(middle_out)[3]))
            for j in range(np.shape(joint3d)[0]):
                temp = 0
                for i in range(np.shape(joint3d)[1]):
                    # Define the 2D point
                    point = np.array([joint3d[j,i,0], joint3d[j,i,1]])

                    # Create a 2D grid of coordinates around the point
                    x, y = np.meshgrid(np.arange(64), np.arange(64))
                    d = np.sqrt((x-point[0])**2 + (y-point[1])**2)

                    # Generate a heatmap with a Gaussian kernel
                    sigma = 0.001
                    #print(np.shape(heatmap[temp,:,:]))
                    heatmap[j, temp, :, :] = (gaussian_filter(d, sigma))
                    heatmap[j, temp, :, :] = np.subtract(np.max(heatmap[j,temp,:,:]), heatmap[j,temp,:,:])
                    temp += 1
            
            #Normalize the heatmap
            #print("SHAPE",np.shape(heatmap))
            #heatmap = np.linalg.norm(heatmap)
            #print("NEWSHAPE", heatmap.shape)
            """mean = np.mean(heatmap, axis=(1,2), keepdims=True)
            std = np.std(heatmap, axis=(1,2), keepdims=True)
            heatmap = np.divide(np.subtract(heatmap, mean), std)"""
            max = np.max(heatmap, axis=(2,3), keepdims=True)
            min = np.min(heatmap, axis=(2,3), keepdims=True)
            #print(min.shape, heatmap.shape)
            #heatmap = np.divide(np.subtract(heatmap, min), np.subtract(max, min))
            heatmap = (heatmap-min)/(max-min)
            # Plot the heatmap
            """if i == 15:
                print(point)
                plt.imshow(heatmap[2,:,:], cmap='hot')
                plt.colorbar()
                plt.show()"""

            #print("HEATMAP",np.shape(heatmap))
            loss = criterion(output, joint3d, middle_out, heatmap)
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
            joint2d,joint3d, image_bv, global_pos_v = data

            image = torch.from_numpy(images_crop(image_bv, global_pos_v, joint3d)).float()
            #print(np.shape(image))
            val_output, middle_out = model(image)
            heatmap = np.zeros((np.shape(joint3d)[0], np.shape(joint3d)[1], np.shape(middle_out)[2],np.shape(middle_out)[3]))
            min_middle = torch.min(middle_out)
            max_middle = torch.max(middle_out)
            middle_out = torch.divide(torch.subtract(middle_out, min_middle), max_middle - min_middle)
            for j in range(np.shape(joint3d)[0]):
                temp = 0
                for i in range(np.shape(joint3d)[1]):
                    # Define the 2D point
                    point = np.array([joint3d[j,i,0], joint3d[j,i,1]])

                    # Create a 2D grid of coordinates around the point
                    x, y = np.meshgrid(np.arange(64), np.arange(64))
                    d = np.sqrt((x-point[0])**2 + (y-point[1])**2)

                    # Generate a heatmap with a Gaussian kernel
                    sigma = 0.001
                    #print(np.shape(heatmap[temp,:,:]))
                    heatmap[j, temp, :, :] = (gaussian_filter(d, sigma))
                    heatmap[j, temp, :, :] = np.subtract(np.max(heatmap[j,temp,:,:]), heatmap[j,temp,:,:])
                    temp += 1
                    
            #Normalize the heatmap
            #print("SHAPE",np.shape(heatmap))
            #heatmap = np.linalg.norm(heatmap)
            #print("NEWSHAPE", heatmap.shape)
            """mean = np.mean(heatmap, axis=(1,2), keepdims=True)
            std = np.std(heatmap, axis=(1,2), keepdims=True)
            heatmap = np.divide(np.subtract(heatmap, mean), std)"""
            max = np.max(heatmap, axis=(2,3), keepdims=True)
            min = np.min(heatmap, axis=(2,3), keepdims=True)
            #print(min.shape, heatmap.shape)
            #heatmap = np.divide(np.subtract(heatmap, min), np.subtract(max, min))
            heatmap = (heatmap-min)/(max-min)
            # Plot the heatmap
            """if i == 15:
                print(point)
                plt.imshow(heatmap[2,:,:], cmap='hot')
                plt.colorbar()
                plt.show()"""
            loss_val = criterion(val_output, joint3d, middle_out, heatmap)
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
        
            x1 = output[0,:, 2].detach().numpy()
            y1 = output[0,:, 0].detach().numpy()
            z1 = -output[0,:, 1].detach().numpy()

            # Creating the 3D plot
            fig = plt.figure()
            ax1 = fig.add_subplot(111, projection='3d')

            # Scatter plot of the data points
            

            # Set labels for the axes
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')

            # Show the plot
            
            x = joint3d[0,:, 2]
            y = joint3d[0,:, 0]
            z = -joint3d[0,:, 1]

            # Creating the 3D plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Scatter plot of the data points
            

            # Set labels for the axes
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax1.scatter(x1, y1, z1, c='b', marker='o')
            ax.scatter(x, y, z, c='b', marker='o')
            #plt.imshow((np.transpose(image[0])))
            #plt.show()

      

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
