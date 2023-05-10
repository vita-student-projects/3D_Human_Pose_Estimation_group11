import torch
import torch.nn as nn

import numpy as np


class MPJPE_Loss(nn.Module):
    def __init__(self):
        super(MPJPE_Loss, self).__init__()

    def forward(self, pred, joint3d, middle_out, joint2d, heatmap, correspondance):
        """
        Compute Mean Per Joint Position Error (MPJPE) between predicted and ground-truth 3D joint positions.

        Args:
            pred: Tensor of shape (batch_size, num_joints, 3), predicted 3D joint positions
            target: Tensor of shape (batch_size, num_joints, 3), ground-truth 3D joint positions

        Returns:
            loss: Tensor of shape (), mean per joint position error
        """

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
            l3D = torch.sum(torch.abs(joint3d[:,:,0] - pred[:,:,0]) + torch.abs(joint3d[:,:,1] - pred[:,:,1]))
        elif np.shape(joint3d)[2] == 3:
            l3D = torch.sum(torch.abs(joint3d - pred))
           

        #L2D
        #If only Human3.6M, stops at 17
        #If MPII remove 7th, 8th and 9th and add 18th
        H_lay = middle_out[0,:np.shape(heatmap)[0], :, :]

        diff = torch.sub(H_lay, torch.from_numpy(heatmap)).detach().numpy()
        L2D = np.sum(np.linalg.norm(diff, axis=(1,2))**2)
        """ diff = torch.sqrt(torch.sum(diff ** 2, dim=2))
        loss = torch.sum(diff) """
        print("L2D",L2D)

        #LHEMlets
        ################################################
        #
        # This create the Tgt. It should be displaced to be done only once in the beginning
        #
        ################################################
        LHEM = 0
        #List of [[parent, child]]
        parent = [[13, 15], [13, 17], [13, 18], [13, 19], [13, 25], [13, 26], [13, 27], [0, 13], [0, 1], [0, 2], [0, 3], [0, 6], [0, 7], [0, 8]]
        for i in range(np.shape(parent)[0]):
            #print("CORRESPONDANCE : First is z, 9 ",correspondance.get(parent[i][0]), correspondance.get(parent[i][1]))

            #Parent and child
            p = parent[i][0]
            c = parent[i][1]
            heatmap_p = heatmap[correspondance.get(p)]
            heatmap_c = heatmap[correspondance.get(c)]

            #joint 17 is not used for H3.6m
            zp = joint3d[0, correspondance.get(p), 2]
            zc = joint3d[0, correspondance.get(c), 2]
            T_GT = self.heatmap_triplets(zp, zc, heatmap_p, heatmap_c)

            #To be added after GT is good
            Lambda_H36m = np.ones(np.shape(T_GT))
            #Lambda_H36m[17] = np.zeros(np.shape(T_GT)[0])
            T_pred = middle_out[0, 17 + 3*i:17 + 3*(i+1), :, :].detach().numpy()
            LHEM += np.sum(np.linalg.norm(np.multiply((T_GT - T_pred), Lambda_H36m), axis=(1,2))) ** 2

        ##############################################################
        Lint = LHEM + L2D
        loss = l3D + 0.05*Lint
        print("LHEM", LHEM)
        print("Lint, L3D", 0.05*Lint, l3D)

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
