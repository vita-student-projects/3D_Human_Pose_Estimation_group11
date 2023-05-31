import cv2
import os
import torch
import sys
import numpy as np
sys.path.append('./')
from table import *
from draw_figure import *
from getActionID import *
from metric_3d import * 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.gridspec as gridspec
import imageio_ffmpeg
import Inference_on_video.dataset_inference_test as dataset_inference_test
from Inference_on_video.dataset_inference_test import H36M
import dataloader
import argparse

import time 
import json
detected = False
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
        # plt.xlim([-1.5, 1.5])
        # plt.ylim([-1.5, 1.5])
        ax.invert_zaxis()
        plt.show()

def draw_plots(joints, img):
    fig = plt.figure( figsize=(19.2 / 2, 10.8 / 2) )
    gs1 = gridspec.GridSpec(1, 2) # 6 rows, 10 columns
    gs1.update(left=0.08, right=0.98,top=0.95,bottom=0.08,wspace=0.05, hspace=0.1)

    font = {'family' : 'serif',  
        'color'  : 'darkred',  
        'weight' : 'normal',  
        'size'   : 10,  
            }
    
    axImg=plt.subplot(gs1[0,0])
    axImg.axis('off')
    # axImg.set_title('Input Image' )#,fontdict=font)

    axPose3d_pred=plt.subplot(gs1[0,1],projection='3d')
    img = np.transpose(img, (1,2,0))
    
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(type(img))
    axImg.imshow(img)
    axImg.axis('off')

    Draw3DSkeleton(joints, axPose3d_pred,JOINT_CONNECTIONS,'Pred_joint3d',fontdict=font,j18_color=JOINT_COLOR_INDEX,image = None)
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
    # max_size = np.max(maxAxis-minAxis) / 2 * 1.1
    
    # ax.set_xlim3d([-max_size + xroot, max_size + xroot])
    # ax.set_ylim3d([-max_size + zroot, max_size + zroot])
    # ax.set_zlim3d([-max_size + yroot, max_size + yroot])
    
    max_size = 130
    ax.set_xlim3d([-max_size , max_size ])
    ax.set_ylim3d([-max_size , max_size ])
    ax.set_zlim3d([-max_size , max_size ])

    if Name is not None :
        ax.set_title(Name,fontdict=fontdict)
    # ax.set_aspect('equal')
def images_crop(images):
    global detected
    original_x = np.shape(images)[2]
    original_y = np.shape(images)[1]
    print(original_x, original_y)
    net = cv2.dnn.readNet("../ckpt/yolov3.weights","../ckpt/yolov3.cfg")
    model_crop = cv2.dnn_DetectionModel(net)
    #Resize into a small square (320,320) to process a fast analysis
    #Scale because the dnn go from 0 to 1 and the pixel value from 0 to 255
    # Calculate the maximum dimension


    max_dim = max(original_x, original_y)

    # Resize the image while maintaining the aspect ratio
    # resized_image = cv2.resize(images[0], (max_dim, max_dim))

    # Create a blank square canvas
    square_image = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)

    # Calculate the padding values
    pad_x = (max_dim - original_x) // 2
    pad_y = (max_dim - original_y) // 2

    # Place the resized image onto the canvas
    square_image[pad_y:pad_y+original_y, pad_x:pad_x+original_x] = images[0]

    model_crop.setInputParams(size=(max_dim,max_dim), scale=1/255)
    print(int(original_x/2), int(original_y/2))
    # image = cv2.resize(images[0],dsize = (int(original_x/2), int(original_y/2)))

    classes = []
    with open("../ckpt/classes.txt", "r") as file_object:
        for class_name in file_object.readlines():
            #To get the good shape of inputs
            class_name = class_name.strip()
            classes.append(class_name)
    print(type(images))
    # image = cv2.resize(images, (320,320))
    res_cropped = np.zeros((256,256,3))
    for i in range(np.shape(images)[0]):
        # image=(np.array(images[i,:,:,:].detach().numpy()))
        # image = image.astype(np.uint8)

        (class_ids, score, bound_boxes) = model_crop.detect(square_image)
        # plt.imshow(np.transpose(images[i,:,:,:]))
        # plt.show()
        print(class_ids, score)

        square_image = square_image.astype(np.float)/255
        print(np.shape(square_image))
        score_lim = 0.98
        people_ids = np.where((class_ids == 0) & (score > score_lim))[0]
        nbr_people = len(people_ids)
        people = np.zeros((nbr_people, 256, 256, 3))
        print("Number of zeros:", nbr_people)
        print("Zero indices:", people_ids)
        counter = 0
        counter2 = 0
        detected = False
        for class_ids, score, bound_boxes in zip(class_ids, score, bound_boxes):

            x, y, w, h = bound_boxes
            #print(x, y, h, w)
            class_name=classes[int(class_ids)]
            if counter < nbr_people:
                # print(classes[int(class_ids)], people_ids[counter], counter, counter2, nbr_people)
                # print("TRUE", np.shape(people))
                if class_name=="person" and counter2 == people_ids[counter] and score > score_lim:

                    detected = True
                    # cv2.putText(square_image, str(class_name)+str(score), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 3, (200, 0, 50), 2)
                    # cv2.rectangle(square_image, (x,y), (x+w,y+h), (200, 0, 50), 3)
                    # cv2.imshow("Frame", square_image)
                    # cv2.waitKey(0)
                    #print(np.shape(image))
                    add = int(h * 0.1)
                    image = np.copy(images[i])
                    original_x = 256
                    original_y = 256
                    if h >= w:
                        diff = int((h-w)/2)
                        low = x - diff - add
                        low = np.clip(low, 0, max_dim)
                        high = x - diff + h + add
                        high = np.clip(high, 0, max_dim)

                        cropped = square_image[np.clip(y-add, 0, max_dim):np.clip(y+h+add, 0, max_dim),low:high,:]
                        
                    else:
                        diff = int((w-h)/2)
                        low = y - diff - add
                        low = np.clip(low, 0, max_dim)
                        high = y - diff + w + add
                        high = np.clip(high, 0, max_dim)
                        cropped = square_image[low:high,np.clip(x-add, 0, max_dim):np.clip(x+w+add, 0, max_dim),:]
                    print(diff, low, high, x,y,h, w)
                    print(np.shape(cropped))
                    res_cropped[:,:,:] = (cv2.resize((cropped), (256,256)))
                    people[counter] = res_cropped
                    counter += 1
            counter2 += 1
    print(np.shape(people))
        # show_skeleton(global_pos)
    # plt.imshow(square_image)
    # plt.show()
    return people, counter

def normalized_to_original(image):
    image_numpy = image.cpu().numpy()
    image_numpy = np.transpose(image_numpy, (0, 2, 3, 1))
    image_numpy = image_numpy * img_std + img_mean
    return image_numpy.astype(np.uint8)

if __name__ == '__main__':
    show_images = False
    parser = argparse.ArgumentParser(description="HEMlets(pose) inference script")
    parser.add_argument("--ckpt_path", type=str,\
                        default="./ckpt/hemlets_h36m_lastest.pth", \
                        help='path to model of the pretrained model')
    parser.add_argument("--video_path", type=str,\
                        default='../data/test_set/test4_cut.mp4', \
                        help='path to a dataset')


    argspar = parser.parse_args()
    print('argspar',argspar)

    from config import config
    from network import Network
    import dataloader
    from model_opr import load_model

    # define network 
    model = Network(config)
    # device = torch.device('cpu')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)


    video_path = argspar.video_path
    checkpoint_path = argspar.ckpt_path

    # load model weights
    load_model(model, checkpoint_path, cpu=not torch.cuda.is_available())
    model.eval()


    #train_dataset = H36M(args.dataset_path, split='val')
    train_dataset = H36M(video_path)
    print("TRAIN DATASET", np.shape(train_dataset))

    test = dataloader.val_loader(train_dataset, config, 1)
    print("TEST", np.shape(test))




    #CREATE JSON
    project_name = "2_3D_Human_Pose_Estimation"
    output_json = []
    for (idx, data) in enumerate(test):
        print("HERE")

        image = data
        print(np.shape(image), type(image))
        image = image.numpy()
        # print(np.max(image[0], axis=(1,2)), np.min(image[0], axis=(1,2)))
        if show_images:
            plt.imshow(image[0])
            plt.show()

        image_new, nbr_people = images_crop(image)
        if detected:
            print("DETECTED")       
            # image_new = cv2.resize(image_new, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)#Important to have an image 256x256!!!
            for k in range(nbr_people):
                image_new[k] = cv2.rotate(image_new[k], cv2.ROTATE_90_COUNTERCLOCKWISE)
                image_new[k] = cv2.flip(image_new[k], 0)
            print(np.shape(image_new))
            # define dataset and dataloader
            print(np.shape(image_new))
            image_new = np.transpose(image_new, (0,2,1,3))
            print(np.shape(image_new))
            # image_new = np.expand_dims(image_new, axis = 0)
            image_new = np.transpose(image_new, (0,3,1,2))
            print("HERE",np.shape(image_new), type(image_new))
            image_new = torch.Tensor(image_new)
            output = model(image_new)
            output[:,:,:2] = (output[:,:,:2] + 0.5)*256.0
            output[:,:,2] *= 128
            print(np.shape(output))
            if show_images:
                for k in range(nbr_people):
                    draw_plots(output[k], image_new[k])
            output[:,:,:] = output[:,:,:] - output[:,0:1,:]

            # output = output.detach().numpy()
            # show_skeleton(output)
            #JSON
            frame_predictions = {
                "frame": idx,
                "predictions": []
            }
            for i in range(nbr_people):
                output_temp = np.array(output[i].detach().numpy()).tolist()
                prediction = {
                    "id": i,
                    "pred": output_temp
                }
                frame_predictions["predictions"].append(prediction)
            output_json.append(frame_predictions)

        else:
            frame_predictions = {
                "frame": idx,
                "predictions": []  # Empty list for now, you can add your predictions here
            }
            for i in range(nbr_people):
                prediction = {
                    "id": i,
                    "pred": []
                }
            frame_predictions["predictions"].append(prediction)
            output_json.append(frame_predictions)

            print("PASS")
            passimage = np.transpose(image, (0,3,1,2))


    # Create the JSON data structure
    data_json = {
        "project": project_name,
        "output": output_json
    }

    # Write the JSON data to a file
    with open("RESULTS.json", "w") as json_file:
        json.dump(data_json, json_file, indent=4)

    print("JSON file created successfully.")