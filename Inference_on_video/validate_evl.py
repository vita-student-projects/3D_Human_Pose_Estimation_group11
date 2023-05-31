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
detected = False

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
    img = ((img * img_std) + img_mean).detach().numpy().astype(np.uint8)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    #print max_size,vals
    if False:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    # Get rid of the ticks and tick labels
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])

    # ax.get_xaxis().set_ticklabels([])
    # ax.get_yaxis().set_ticklabels([])
    # ax.set_zticklabels([])
    # ax.set_aspect('equal')



    # px = np.arange(-max_size + xroot, max_size + xroot, 3)   
    # pz = np.arange(-max_size + zroot, max_size + zroot, 3)     
    # px, pz = np.meshgrid(px,pz)   

    # py = np.zeros((px.shape[0],px.shape[1]),dtype=float)
    # py[:,:]=vals[:,1].max()

    # #print px.shape,pz.shape,py.shape
    # ax.axis('off')
    # surf = ax.plot_surface(px, pz, py,color='gray',alpha=0.5)   

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
        square_image = square_image.astype(np.float)/255
        print(np.shape(square_image))

        for class_ids, score, bound_boxes in zip(class_ids, score, bound_boxes):
            x, y, w, h = bound_boxes
            #print(x, y, h, w)
            class_name=classes[int(class_ids)]
            detected = False
            if class_name=="person":
                detected = True
                # cv2.putText(square_image, str(class_name)+str(score), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 3, (200, 0, 50), 2)
                # cv2.rectangle(square_image, (x,y), (x+w,y+h), (200, 0, 50), 3)
                # cv2.imshow("Frame", square_image)
                # cv2.waitKey(0)
                #print(np.shape(image))
                add = 20
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
                break
       
        # show_skeleton(global_pos)
    # plt.imshow(square_image)
    # plt.show()
    return res_cropped

def normalized_to_original(image):
    image_numpy = image.cpu().numpy()
    image_numpy = np.transpose(image_numpy, (0, 2, 3, 1))
    image_numpy = image_numpy * img_std + img_mean
    return image_numpy.astype(np.uint8)

def validate(model, val_loader, device,  subject=9,visualize = False, First = False):

    if visualize:
        if First:
            fig = plt.figure(figsize=(40, 60) )
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

        #axPose3d_gt=plt.subplot(gs1[0,1],projection='3d')
        axPose3d_pred=plt.subplot(gs1[0,1],projection='3d')

        #ax_mpjpe = plt.subplot(gs1[1,:])
    
        vw_pure = None
        vw_path = './hemlets.mp4'
    
        protocol_1_list = []
        protocol_2_list = []


    

    seqList = [11]
    seqJsonDict = {}
    for seq in seqList:
        seqJsonDict[seq] =  LoadSeqJsonDict(rootPath = '../data/',subject=seq)

    # define action wise protocol dict
    action_wise_error_dict = {}
    for i in range(15):
        action_wise_error_dict[i] = [0.0, 0.0, 0.0]
    
    # result_pics_path = './pics/'

    # if os.path.exists(result_pics_path) is False:
    #     os.mkdir(result_pics_path)

    N_viz = val_loader.__len__() 
    idx_list = []
    for idx, data in enumerate(val_loader):
        if idx >= N_viz:
            break
        
        image,image_flip, trans, camid, joint3d, joint3d_camera,  root, name = data

        image = image.to(device)
        # for flip test to improve the accuracy of 3d pose prediction 
        image_flip = image_flip.to(device)

        joint3d = joint3d.to(device)

        # start_time = time.time()

        with torch.no_grad():
            pred_joint3d = model(image,val=True)
            pred_joint3d_flip = model(image_flip,val=True)

        
        pred_joint3d_flip_numpy = pred_joint3d_flip.cpu().numpy()
        pred_joint3d_numpy = pred_joint3d.cpu().numpy()

        # normalized (0-1) 3d joints
        gt_joint3d_numpy = joint3d.cpu().numpy()
        # camera space 3d joints
        joint3d_camera_numpy = joint3d_camera.cpu().numpy()

        # sequence encodered information [seqID,CamID,_,video_number]
        camid_numpy = camid.numpy()
        # transform matrix between the croppred (256*256) and org image (1000*1000)
        trans_numpy = trans.numpy()
        # abs joint root 
        joint_root_numpy = [None] 

        

        # calculate the MPJPE(Protocol #1) and MPJPE(Protocol #2)
        actionID, protocol_1,protocol_2,vedioName,gt_crop_3d_joint,pred_crop_3d_joint= \
                    eval_metric(pred_joint3d_numpy,pred_joint3d_flip_numpy,gt_joint3d_numpy,camid_numpy,\
                    trans_numpy,joint_root_numpy,joint3d_camera_numpy,seqJsonDict = seqJsonDict,debug = False,return_viz_joints=True)
        print('actionID, protocol_1,protocol_2,vedioName',actions[actionID], protocol_1,protocol_2,vedioName)
        # inference_time = time.time()-start_time


        if visualize:
            # convert image tensor to image numpy
            image_batch = normalized_to_original(image)

            # buffered the protocol #1 & #2 Error (in mm) for every frame.
            protocol_1_list.append(protocol_1)
            protocol_2_list.append(protocol_2)
            idx_list.append(idx)

            # start_img_time = time.time()

            img = image_batch[0]
            # draw gt 2d joint on the image  
            #image_draw = drawSkeleton(img.copy(),gt_crop_3d_joint,JOINT_CONNECTIONS,JOINT_COLOR_INDEX)
            axImg.imshow(img)
            axImg.axis('off')
            axImg.set_title('S_11: ({})'.format(vedioName),fontdict=font)
            # img_time = time.time() - start_img_time

            # start_skel_time = time.time()
            # draw 3d joints
            
            #Draw3DSkeleton(gt_crop_3d_joint,axPose3d_gt,JOINT_CONNECTIONS,'GT_joint3d',fontdict=font,j18_color=JOINT_COLOR_INDEX,image = None)
            Draw3DSkeleton(pred_crop_3d_joint,axPose3d_pred,JOINT_CONNECTIONS,'Pred_joint3d',fontdict=font,j18_color=JOINT_COLOR_INDEX,image = None)
            # skel_time = time.time() - start_skel_time
            # plot the cruve of Protocol #1 and Protocol #2

            # start_line_time = time.time()

            #ax_mpjpe.plot(idx_list[-3:],protocol_1_list[-3:],color='b',label='Protocol #1')
            #ax_mpjpe.plot(idx_list[-3:],protocol_2_list[-3:],color='g',label='Protocol #2')

            # ax_mpjpe.plot([idx-0.5,idx+0.5],[protocol_1_list[-1],protocol_1_list[-1]],color='b',label='Protocol #1')
            # ax_mpjpe.plot([idx-0.5,idx+0.5],[protocol_2_list[-1],protocol_2_list[-1]],color='g',label='Protocol #2')

            # line_time = time.time() - start_line_time

            # start_save_time = time.time()

            # add legend annotations
            
            """ plt.legend(loc = 'upper right',edgecolor='blue')
            ax_mpjpe.set_ylabel('Pose Error (mm)',fontsize = 10)
            ax_mpjpe.set_xlabel('Frame',fontsize = 10)

            ax_mpjpe.set_xlim(0,N_viz)
            ax_mpjpe.set_ylim(7,65)

            if idx >= N_viz-1:
                ax_mpjpe.plot(idx_list,protocol_1_list,color='b',label='Protocol #1')
                ax_mpjpe.plot(idx_list,protocol_2_list,color='g',label='Protocol #2') """

            plt.draw()             
            plt.pause(0.01)
            
            """ if idx == N_viz-1:
                input('check') """
            # plt.savefig('{}/{}.png'.format(result_pics_path,str(idx).zfill(5)))

            plt.savefig('./temp.png')
            img_viz = cv2.imread('./temp.png')
            axImg.cla()
            #axPose3d_gt.cla()
            axPose3d_pred.cla()
            #ax_mpjpe.cla()
            # save_time = time.time() - start_save_time

            # print('time',inference_time,img_time,skel_time,line_time,save_time)
            #write the rendered frame into a video
            if vw_pure is None:# initialize the video writer 
                wSize = (img_viz.shape[1],img_viz.shape[0])
                fps = 25

                vw_pure = imageio_ffmpeg.write_frames(vw_path,
                                                        wSize,
                                                        fps=fps,
                                                        quality=8,
                                                        
                                                        ffmpeg_timeout=0,       # libx264 in default
                                                        macro_block_size=1) 
                vw_pure.send( None )

            # convert BGR to RGB 
            vw_pure.send(img_viz[:,:,::-1].copy())

    
        # cache the action-wise error 
        action_wise_error_dict[actionID][0] += protocol_1
        action_wise_error_dict[actionID][1] += protocol_2
        action_wise_error_dict[actionID][2] += 1

    if visualize:
        vw_pure.close()
    

    for k,_ in action_wise_error_dict.items():
        if action_wise_error_dict[k][2]>0:
            print(actions[k],action_wise_error_dict[k][0]/action_wise_error_dict[k][2],action_wise_error_dict[k][1]/action_wise_error_dict[k][2])
    
    print("END")

if __name__ == '__main__':
    
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

    for (idx, data) in enumerate(test):
        print("HERE")

        image = data
        print(np.shape(image), type(image))
        image = image.numpy()
        # print(np.max(image[0], axis=(1,2)), np.min(image[0], axis=(1,2)))
        plt.imshow(image[0])
        plt.show()

        image_new = images_crop(image)
        if detected:
            print("DETECTED")       
            image_new = cv2.resize(image_new, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)#Important to have an image 256x256!!!
            image_new = cv2.rotate(image_new, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image_new = cv2.flip(image_new, 0)
            # define dataset and dataloader
            print(np.shape(image_new))
            image_new = np.transpose(image_new)
            image_new=image_new[[2,1,0], :, :]
            image_new=image_new[[0,1,2], :, :]
            print(np.shape(image_new))
            image_new = np.expand_dims(image_new, axis = 0)
            print("HERE",np.shape(image_new), type(image_new))
            image_new = torch.Tensor(image_new)
            output = model(image_new)
            output[:,:,:2] = (output[:,:,:2] + 0.5)*256.0
            output[:,:,2] *= 128
            draw_plots(output, image_new[0])
        else:
            print("PASS")
            passimage = np.transpose(image, (0,3,1,2))
