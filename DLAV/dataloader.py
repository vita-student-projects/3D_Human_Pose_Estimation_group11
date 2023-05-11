import math

import torch

from sklearn import shuffle



def val_loader(dataset, config, data_ratio, validation_ratio, test_ratio, batch_size):
    num_data = len(dataset)
    print(dataset)
    dataset = shuffle(dataset)

    data_length = int(data_ratio * num_data)
    print("data_length",data_length)
    test_split = int(test_ratio * data_length)
    print("test_split",test_split)
    train_val_split = int((1-test_ratio) * data_length)
    print("train_val_split",train_val_split)
    train_split = int((1 - validation_ratio) * train_val_split)
    print("train_split", train_split)
    validation_split = train_val_split - train_split
    print("validation_split", validation_split)

    # Create train/validation/test subsets using Subset class
    train_val_subset = torch.utils.data.Subset(dataset, range(train_val_split))
    test_subset = torch.utils.data.Subset(dataset, range(data_length - test_split, data_length))

    # Split train_val_subset into train and validation subsets
    train_split = int((1 - validation_ratio) * len(train_val_subset))
    train_subset = torch.utils.data.Subset(train_val_subset, range(train_split))
    val_subset = torch.utils.data.Subset(train_val_subset, range(train_split, len(train_val_subset)))

    print("train_subset",len(train_subset))
    sampler = torch.utils.data.sampler.SequentialSampler(train_subset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=False)

    train_loader = torch.utils.data.DataLoader(train_subset, num_workers=config.VAL.NUM_WORKERS, batch_sampler=batch_sampler)

    sampler_val = torch.utils.data.sampler.SequentialSampler(val_subset)
    batch_sampler_val = torch.utils.data.sampler.BatchSampler(sampler_val, validation_split, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_subset, num_workers=config.VAL.NUM_WORKERS, batch_sampler=batch_sampler_val)

    return train_loader, val_loader, test_subset


if __name__ == '__main__':
    from easydict import EasyDict as edict
    from exps.baseline.config import config


    from dataset import get_train_dataset, get_val_dataset
    dataset = get_train_dataset(config)
    # dataset = get_val_dataset(config)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    torch.distributed.init_process_group(backend="nccl", init_method='env://')

    # loader = train_loader(dataset, config)
    loader = val_loader(dataset, config, 0, 2)

    iter_loader = iter(loader)
    if args.local_rank == 0:
        lr, hr = iter_loader.next()
        print(lr.size(), hr.size())
