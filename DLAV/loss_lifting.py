import torch
import torch.nn as nn

import numpy as np
import HEMlets.config
from scipy.stats import multivariate_normal

class MPJPE_Loss(nn.Module):
    def __init__(self):
        super(MPJPE_Loss, self).__init__()

    def forward(self, pred, joints, joint2d):
        loss = nn.MSELoss()
        mse_loss = loss(pred, joints)
        
        return mse_loss