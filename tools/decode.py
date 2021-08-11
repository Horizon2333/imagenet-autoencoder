#!/usr/bin/env python

import argparse

import matplotlib.pyplot as plt

import torch

from torchvision.transforms import transforms

import sys
sys.path.append("./")

import utils
import models.builer as builder

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Trainer for auto encoder')
    parser.add_argument('--arch', default='vgg16', type=str, 
                        help='backbone architechture')
    parser.add_argument('--resume', type=str)      
    
    args = parser.parse_args()

    args.parallel = 0
    args.batch_size = 1
    args.workers = 0

    return args

def random_sample(arch):

    if arch in ["vgg11", "vgg13", "vgg16", "vgg19", "resnet18", "resnet34"]:
        return torch.randn((1,512,7,7))
    elif arch in ["resnet50", "resnet101", "resnet152"]:
        return torch.randn((1,2048,7,7))
    else:
        raise NotImplementedError("Do not have implemention except VGG and ResNet")

def main(args):
    print('=> torch version : {}'.format(torch.__version__))

    utils.init_seeds(1, cuda_deterministic=False)

    print('=> modeling the network ...')
    model = builder.BuildAutoEncoder(args)     
    total_params = sum(p.numel() for p in model.parameters())
    print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))

    print('=> loading pth from {} ...'.format(args.resume))
    utils.load_dict(args.resume, model)

    trans = transforms.ToPILImage()

    plt.figure(figsize=(16, 9))

    model.eval()
    print('=> Genarating ...')
    with torch.no_grad():
        for i in range(128):
            
            input = random_sample(arch=args.arch).cuda()

            output = model.module.decoder(input)

            output = trans(output.squeeze().cpu())

            plt.subplot(8,16,i+1, xticks=[], yticks=[])
            plt.imshow(output)

    plt.savefig('figs/generation.jpg')

if __name__ == '__main__':

    args = get_args()

    main(args)


