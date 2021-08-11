#!/usr/bin/env python

import argparse

from PIL import Image

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
    parser = argparse.ArgumentParser(description='Encoder for auto encoder')
    parser.add_argument('--arch', default='vgg16', type=str, 
                        help='backbone architechture')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--img_path',type=str)              
    
    args = parser.parse_args()

    args.parallel = 0
    args.batch_size = 1
    args.workers = 0

    return args

def encode(model, img):

    with torch.no_grad():

        code = model.module.encoder(img).cpu().numpy()

    return code

def main(args):
    print('=> torch version : {}'.format(torch.__version__))

    utils.init_seeds(1, cuda_deterministic=False)

    print('=> modeling the network ...')
    model = builder.BuildAutoEncoder(args)     
    total_params = sum(p.numel() for p in model.parameters())
    print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))

    print('=> loading pth from {} ...'.format(args.resume))
    utils.load_dict(args.resume, model)
    
    trans = transforms.Compose([
                    transforms.Resize(256),                   
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                  ])

    img = Image.open(args.img_path).convert("RGB")

    img = trans(img).unsqueeze(0).cuda()

    model.eval()

    code = encode(model, img)

    print(code.shape)

    # To do : any other postprocessing

if __name__ == '__main__':

    args = get_args()

    main(args)


