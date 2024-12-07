import os
import json
import numpy as np
import torch
import torchvision
from torchvision.transforms import v2 

def get_transforms():
    # Set train_datatransforms
    s = 1
    color_jitter = v2.ColorJitter(
            0.9 * s, 0.9 * s, 0.9 * s, 0.1 * s)
    flip = v2.RandomHorizontalFlip()
    train_transforms = v2.Compose(
                [
                    v2.RandomResizedCrop(size=32),
                    v2.RandomApply([flip], p=0.5),
                    v2.RandomApply([color_jitter], p=0.9),
                    v2.RandomGrayscale(p=0.1),                   
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    unitmem_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    )
    test_transforms = v2.Compose(
        [
            v2.Resize(32),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    return train_transforms, unitmem_transforms, test_transforms

def init_dataset(trainer_or_unitmem, args):
    
    train_transforms, unitmem_transforms, test_transforms = get_transforms()
    
    # Set data based on pruning method
    print("Current working directory:", os.getcwd(), flush = True)
    data_dir = "ayeshas_code/data"
    datasets = {
        'cifar_10': ('cifar-10-python', torchvision.datasets.CIFAR10),
        'cifar_100': ('cifar-100-python', torchvision.datasets.CIFAR100)
    }

    datapath = os.path.join(data_dir, datasets[args.data][0])
    dataset_class = datasets[args.data][1]

    # Dataset loading based on type
    if trainer_or_unitmem == "trainer":   
        train_dataset = dataset_class(root=datapath, train=True, 
                                    download=True, transform=train_transforms)
    elif trainer_or_unitmem == "unitmem":
        train_dataset = dataset_class(root=datapath, train=True, 
                                    download=True, transform=unitmem_transforms)

    test_dataset = dataset_class(root=datapath, train=True, 
                            download=True, transform=test_transforms)

    return train_dataset, test_dataset


    # # Wrap the original datasets with IndexedDataset
    # train_dataset = IndexedDataset(org_train_dataset)
    # test_dataset = IndexedDataset(org_test_dataset)