import torch
import torch.nn as nn

class MPJPE_Loss(nn.Module):
    def __init__(self):
        super(MPJPE_Loss, self).__init__()

    def forward(self, pred, target):
        """
        Compute Mean Per Joint Position Error (MPJPE) between predicted and ground-truth 3D joint positions.

        Args:
            pred: Tensor of shape (batch_size, num_joints, 3), predicted 3D joint positions
            target: Tensor of shape (batch_size, num_joints, 3), ground-truth 3D joint positions

        Returns:
            loss: Tensor of shape (), mean per joint position error
        """

        assert pred.shape == target.shape, "Predicted and target shapes do not match"

        num_joints = pred.shape[1]
        diff = pred - target
        diff = torch.sqrt(torch.sum(diff ** 2, dim=2))
        loss = torch.sum(diff) / (num_joints * pred.shape[0])

        return loss