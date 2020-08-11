from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import numpy as np
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
import PIL


class SubsetSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def get_loader(root, test_batch, train_batch, valid_size=0, num_workers=16, random_seed=12345):
    indices = list(range(50000))
    np.random.seed(random_seed)
    np.random.shuffle(indices)            
    test_idx, valid_idx = indices[valid_size:], indices[:valid_size]
    
    if train_batch > 0:
        transform_train = transforms.Compose([
          transforms.RandomResizedCrop(224, interpolation=PIL.Image.BICUBIC),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                              std = [ 0.229, 0.224, 0.225 ]),
        ])


        train_dataset = datasets.ImageFolder(os.path.join(root, "train"), transform_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch, 
            shuffle=True, num_workers=num_workers, pin_memory=True)
    else:
        train_loader = None
    
    if valid_size > 0:
        assert(test_batch > 0, "validation set follows the batch size of test set, which is 0")
        transform_valid = transforms.Compose([
            transforms.Resize(256, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                    std = [ 0.229, 0.224, 0.225 ]),
        ])

        valid_dataset = datasets.ImageFolder(os.path.join(root, "val_pt"), transform_valid)
        valid_sampler =  SubsetRandomSampler(valid_idx) 
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=test_batch, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=False, drop_last=False)
    else:
        valid_loader = None    
    
    if test_batch > 0:
        transform_test = transforms.Compose([
            transforms.Resize(256, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                    std = [ 0.229, 0.224, 0.225 ]),
        ])

        test_dataset = datasets.ImageFolder(os.path.join(root, "val_pt"), transform_test)
        test_sampler = SubsetSampler(test_idx)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch, sampler=test_sampler,
            num_workers=num_workers, pin_memory=True, drop_last=False)

    else:
        test_loader = None

    return test_loader, train_loader, valid_loader
