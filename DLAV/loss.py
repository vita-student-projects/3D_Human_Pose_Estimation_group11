import torch
import torch.nn as nn

import numpy as np
import HEMlets.config
from scipy.stats import multivariate_normal

class MPJPE_Loss(nn.Module):
    def __init__(self):
        super(MPJPE_Loss, self).__init__()

    def forward(self, pred, joints, middle_out, joint2d):
        L2D = 0
        LHEM = 0
        l3D = 0
        
        joints = torch.Tensor(joints)
        pred = torch.Tensor(pred)
        joints = joints - joints[:, 0, :].unsqueeze(1)
        pred = pred - pred[:, 0, :].unsqueeze(1)
        
        if pred.requires_grad:
            print("GRAD")

        #HEMlets loss

        #L3D_lambda
        if joints.size()[2] == 2:
            l3D = torch.mean(torch.abs(joints[:,:,0] - pred[:,:,0]) + torch.abs(joints[:,:,1] - pred[:,:,1]))
        elif joints.size()[2] == 3:
            #Loss from paper
            l3D = torch.mean(torch.sum(torch.abs(joints - pred[:,:17,:]), axis = (1,2)))
           
        #L2D
        #If only Human3.6M, stops at 17
        #If MPII remove 7th, 8th and 9th and add 18th
        H_lay = middle_out[:,:joints.size()[1], :, :]
        heatmap_2d = self.create_heatmap(joint2d, np.shape(H_lay)[2])      
        diff = torch.sub(H_lay, torch.from_numpy(heatmap_2d)).detach().numpy()
        L2D = np.mean(np.sum(np.linalg.norm(diff, axis=(2,3))**2, axis = 1))
 
        print("L2D",L2D)

        #LHEMlets
        LHEM = np.zeros(np.shape(heatmap_2d)[0])
        #List of [[parent, child]]
        parent = [[13, 15], [13, 17], [13, 18], [13, 19], [13, 25], [13, 26], [13, 27], [0, 13], [0, 1], [0, 2], [0, 3], [0, 6], [0, 7], [0, 8]]
        parent = [[0,1], [1,2],[2,3],[0,4],[4,5],[5,6],[0,8],[8,14],[14,15],[15,16],[8,11],[11,12],[12,13],[8,10]]
        for j in range(np.shape(heatmap_2d)[0]):
            for i in range(np.shape(parent)[0]):

                #Parent and child
                p = parent[i][0]
                c = parent[i][1]
                heatmap_p = heatmap_2d[j,(p)]
                heatmap_c = heatmap_2d[j,(c)]

                #joint 17 is not used for H3.6m
                zp = joints[j, (p), 2]
                zc = joints[j, (c), 2]
                T_GT = self.heatmap_triplets(zp, zc, heatmap_p, heatmap_c)

                #To be added after GT is good
                Lambda_H36m = np.ones(np.shape(T_GT))
                #Lambda_H36m[17] = np.zeros(np.shape(T_GT)[0])
                T_pred = middle_out[j, 17 + 3*i:17 + 3*(i+1), :, :].detach().numpy()
                LHEM[j] += np.sum(np.linalg.norm(np.multiply((T_GT - T_pred), Lambda_H36m), axis=(1,2))) ** 2
        ##############################################################
        LHEM = np.mean(LHEM)
        print("LHEM",LHEM)
        Lint = 0.005*(LHEM + L2D)
        print("Lint",Lint)
        loss = l3D + Lint
        print("l3D",l3D.item())
        print("loss", loss.item())
        MPJPE_Val=self.MPJPE(joints, pred)
        return loss , MPJPE_Val
    
    def heatmap_triplets(self, zp, zc, heatmap_p, heatmap_c):
        #Check if the child is behind or ahead
        epsilon = 0.1

        if (zp - zc) > epsilon:
            r_zp_zc = 1
        elif np.abs(zp - zc) < epsilon:
            r_zp_zc = 0
        else:
            r_zp_zc = -1

        T_zeros = np.zeros(np.shape(heatmap_c))

        #T is negative, zero and positive polarity heatmap
        if r_zp_zc == 1:
            return np.stack((T_zeros, heatmap_p, heatmap_c))
        elif r_zp_zc == 0:
            zero_polarity = heatmap_p + heatmap_c
            return np.stack((T_zeros, zero_polarity, T_zeros))
        elif r_zp_zc == -1:
            return np.stack((heatmap_c, heatmap_p, T_zeros))
        
    def create_heatmap(self, joints, middle_out_size):
        heatmap = np.zeros((np.shape(joints)[0], np.shape(joints)[1], middle_out_size, middle_out_size))
        for l in range(np.shape(joints)[0]):
            for k in range(np.shape(joints)[1]):
                # Define the 2D point
                x, y = np.array([joints[l,k,0], joints[l,k,1]])*64
                
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

    def MPJPE(self, gnd_truth, estimation):
            
        gnd_truth=gnd_truth.detach().numpy()
        estimation=estimation.detach().numpy()
    
        MPJPE=np.average(np.sqrt(np.sum(np.square(gnd_truth[:,0:17,:] - estimation[:,0:17,:]),axis=-1))) 
        return MPJPE