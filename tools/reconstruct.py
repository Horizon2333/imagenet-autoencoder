#!/usr/bin/env python

import argparse

import matplotlib.pyplot as plt

import torch

from torchvision.transforms import transforms

import sys
sys.path.append("./")

import utils
import models.builer as builder
import dataloader

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Trainer for auto encoder')
    parser.add_argument('--arch', default='vgg16', type=str, 
                        help='backbone architechture')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--val_list', type=str)              
    
    args = parser.parse_args()

    args.parallel = 0
    args.batch_size = 1
    args.workers = 0

    return args

def main(args):
    print('=> torch version : {}'.format(torch.__version__))

    utils.init_seeds(1, cuda_deterministic=False)

    print('=> modeling the network ...')
    model = builder.BuildAutoEncoder(args)     
    total_params = sum(p.numel() for p in model.parameters())
    print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))

    print('=> loading pth from {} ...'.format(args.resume))
    utils.load_dict(args.resume, model)
    
    print('=> building the dataloader ...')
    train_loader = dataloader.val_loader(args)

    plt.figure(figsize=(16, 9))

    model.eval()
    print('=> reconstructing ...')
    with torch.no_grad():
        for i, (input, target) in enumerate(train_loader):
            
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(input)

            input = transforms.ToPILImage()(input.squeeze().cpu())
            output = transforms.ToPILImage()(output.squeeze().cpu())

            plt.subplot(8,16,2*i+1, xticks=[], yticks=[])
            plt.imshow(input)

            plt.subplot(8,16,2*i+2, xticks=[], yticks=[])
            plt.imshow(output)

            if i == 63:
                break

    plt.savefig('figs/reconstruction.jpg')

if __name__ == '__main__':

    args = get_args()

    main(args)


