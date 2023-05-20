import torch
import torch.nn as nn

import numpy as np
import HEMlets.config

class MPJPE_Loss(nn.Module):
    def __init__(self):
        super(MPJPE_Loss, self).__init__()

    def forward(self, pred, joint3d, middle_out, heatmap):
        """
        Compute Mean Per Joint Position Error (MPJPE) between predicted and ground-truth 3D joint positions.

        Args:
            pred: Tensor of shape (batch_size, num_joints, 3), predicted 3D joint positions
            target: Tensor of shape (batch_size, num_joints, 3), ground-truth 3D joint positions

        Returns:
            loss: Tensor of shape (), mean per joint position error
        """
        L2D = 0
        LHEM = 0
        l3D = 0
        #assert pred.shape == joint3d.shape, "Predicted and target shapes do not match"

        num_joints = pred.shape[1]
        #print("PRED", np.shape(pred))
        #basic MPJPE
        """ diff = pred - target
        diff = torch.sqrt(torch.sum(diff ** 2, dim=2))
        loss = torch.sum(diff) / (num_joints * pred.shape[0]) """

        #HEMlets loss

        #Should add losses using this layer (fb_map_feature) (42 x 64 x 64) output of HEMlets
        #print(np.shape(joint3d)[2])

        #L3D_lambda
        #Change the loss depending if the data are in 2D or 3D
        if np.shape(joint3d)[2] == 2:
            l3D = torch.mean(torch.abs(joint3d[:,:,0] - pred[:,:,0]) + torch.abs(joint3d[:,:,1] - pred[:,:,1]))
        elif np.shape(joint3d)[2] == 3:
            #Loss from paper
            l3D = torch.sum(torch.mean(torch.abs(joint3d - pred[:,:17,:]), axis = (0,2)))
            
            #CHAT
            distance = torch.norm(pred[:,:17,:] - joint3d, dim=-1)
            l3D = torch.mean(torch.mean(distance, dim=1), dim=0)


            l3D = torch.sum(torch.mean(torch.sum(torch.abs(joint3d - pred[:,:17,:]), axis = (2)), axis = 0))

        #L2D
        #If only Human3.6M, stops at 17
        #If MPII remove 7th, 8th and 9th and add 18th
        #print("middle_out",np.shape(middle_out))
        H_lay = middle_out[:,:np.shape(heatmap)[1], :, :]
        #print(H_lay.shape)

        diff = torch.sub(H_lay, torch.from_numpy(heatmap)).detach().numpy()
        L2D = np.mean(np.sum(np.linalg.norm(diff, axis=(2,3))**2, axis =1))
        """ diff = torch.sqrt(torch.sum(diff ** 2, dim=2))
        loss = torch.sum(diff) """
        print("L2D",L2D)

        #LHEMlets
        ################################################
        #
        # This create the Tgt. It should be displaced to be done only once in the beginning
        #
        ################################################
        LHEM = np.zeros(np.shape(heatmap)[0])
        #List of [[parent, child]]
        parent = [[13, 15], [13, 17], [13, 18], [13, 19], [13, 25], [13, 26], [13, 27], [0, 13], [0, 1], [0, 2], [0, 3], [0, 6], [0, 7], [0, 8]]
        parent = [[0,1], [1,2],[2,3],[0,4],[4,5],[5,6],[0,8],[8,14],[14,15],[15,16],[8,11],[11,12],[12,13],[8,10]]
        for j in range(np.shape(heatmap)[0]):
            for i in range(np.shape(parent)[0]):
                #print("CORRESPONDANCE : First is z, 9 ",correspondance.get(parent[i][0]), correspondance.get(parent[i][1]))

                #Parent and child
                p = parent[i][0]
                c = parent[i][1]
                heatmap_p = heatmap[j,(p)]
                heatmap_c = heatmap[j,(c)]

                #joint 17 is not used for H3.6m
                zp = joint3d[j, (p), 2]
                zc = joint3d[j, (c), 2]
                T_GT = self.heatmap_triplets(zp, zc, heatmap_p, heatmap_c)

                #To be added after GT is good
                Lambda_H36m = np.ones(np.shape(T_GT))
                #Lambda_H36m[17] = np.zeros(np.shape(T_GT)[0])
                T_pred = middle_out[j, 17 + 3*i:17 + 3*(i+1), :, :].detach().numpy()
                LHEM[j] += np.sum(np.linalg.norm(np.multiply((T_GT - T_pred), Lambda_H36m), axis=(1,2))) ** 2
        ##############################################################
        Lint = np.mean(LHEM)/20 + L2D
        loss = l3D + 0.0*Lint
        print("LHEM", np.mean(LHEM)/20)
        print("Lint, L3D, loss", Lint, l3D.item(), loss.item())

        return loss
    
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

