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
from HEMlets.network import Network
import HEMlets.dataloader as dataloader
from HEMlets.model_opr import load_model
from dataset import H36M
from HEMlets.getActionID import LoadSeqJsonDict

def main(args):
    # Set up dataset and data loader
    tiny_dataset = '../data/S11/S_11_C_4_1.h5'
    train_dataset = H36M(h5_path = tiny_dataset, split='val')
    train_loader = dataloader.val_loader(train_dataset, config, 0, 1)

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
        print("HEEERE",np.shape(enumerate(train_loader)))
        count = 0
        for (idx, data) in enumerate(train_loader):
            #data, target = data.to(device), target.to(device)
            print("SHAPE", np.shape(idx))
            print("SHAPE", np.shape(data[0])) #1, 3, 256, 256
            

            image,image_flip, trans, camid, joint3d, joint3d_camera,  root, name = data
            print("JOINT", joint3d)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, joint3d)
            loss.backward()
            optimizer.step()

            print('Train Epoch: {} [(%)]\tLoss: {:.6f}'.format(
                    epoch, loss.item()))

            """ if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item())) """
            if count == 1:
                break
            count += 1

        # Save model checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_dir = os.path.join(args.checkpoint_dir, 'epoch_{}.pt'.format(epoch))
            torch.save(model.state_dict(), checkpoint_dir)

      

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
    args = parser.parse_args()

    main(args)
