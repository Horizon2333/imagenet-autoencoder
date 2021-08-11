#!/usr/bin/env python

import random

from PIL import Image, ImageFilter

import torch
import torch.utils.data as data

from torchvision.transforms import transforms

class ImageDataset(data.Dataset):
    def __init__(self, ann_file, transform=None):
        self.ann_file = ann_file
        self.transform = transform
        self.init()

    def init(self):
        
        self.im_names = []
        self.targets = []
        with open(self.ann_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split(' ')
                self.im_names.append(data[0])
                self.targets.append(int(data[1]))

    def __getitem__(self, index):
        im_name = self.im_names[index]
        target = self.targets[index]

        img = Image.open(im_name).convert('RGB') 
        if img is None:
            print(im_name)
        
        img = self.transform(img)

        return img, img

    def __len__(self):
        return len(self.im_names)

def train_loader(args):

    # [NO] do not use normalize here cause it's very hard to converge
    # [NO] do not use colorjitter cause it lead to performance drop in both train set and val set

    # [?] guassian blur will lead to a significantly drop in train loss while val loss remain the same

    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        #transforms.RandomGrayscale(p=0.2),
        #transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]

    train_trans = transforms.Compose(augmentation)

    train_dataset = ImageDataset(args.train_list, transform=train_trans)   
    
    if args.parallel == 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                            train_dataset,
                            rank=args.rank,
                            num_replicas=args.world_size,
                            shuffle=True)         
    else:  
        train_sampler = None    

    train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    shuffle=(train_sampler is None),
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    pin_memory=True,
                    sampler=train_sampler,
                    drop_last=(train_sampler is None))

    return train_loader

def val_loader(args):

    val_trans = transforms.Compose([
                    transforms.Resize(256),                   
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                  ])

    val_dataset = ImageDataset(args.val_list, transform=val_trans)      

    val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    shuffle=False,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    pin_memory=True)

    return val_loader 

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x