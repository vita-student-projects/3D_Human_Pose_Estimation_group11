import math

import torch
import numpy as np

def val_loader(dataset, config, batch_size):
    print("DATASET",np.shape(dataset))
    num_data = len(dataset)
    print("NUM_DATA",num_data)
    print(dataset)
    sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=False)

    test = torch.utils.data.DataLoader(dataset, num_workers=config.VAL.NUM_WORKERS, batch_sampler=batch_sampler)

    return test

