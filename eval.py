#!/usr/bin/env python

import os
import time
import argparse

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import utils
import models.builer as builder
import dataloader

def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Evaluate for auto encoder')
    parser.add_argument('--arch', default='vgg16', type=str, 
                        help='backbone architechture')
    parser.add_argument('--val_list', type=str)
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', 
                        help='number of data loading workers (default: 0)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')

    parser.add_argument('-p', '--print-freq', default=20, type=int,
                        metavar='N', help='print frequency (default: 10)')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--folder', type=str)   
    parser.add_argument('--start_epoch', default=0, type=int)                                 
    parser.add_argument('--epochs', default=100, type=int) 

    args = parser.parse_args()

    args.parallel = 0

    return args

def main(args):
    print('=> torch version : {}'.format(torch.__version__))
    ngpus_per_node = torch.cuda.device_count()
    print('=> ngpus : {}'.format(ngpus_per_node))

    utils.init_seeds(1, cuda_deterministic=False)
    
    print('=> modeling the network ...')
    model = builder.BuildAutoEncoder(args)      
    total_params = sum(p.numel() for p in model.parameters())
    print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))
    
    print('=> building the dataloader ...')
    val_loader = dataloader.val_loader(args)

    print('=> building the criterion ...')
    criterion = nn.MSELoss()

    print('=> starting evaluating engine ...')
    if args.folder:
        best_loss = None
        best_epoch = 1
        losses = []
        for epoch in range(args.start_epoch, args.epochs):
            print()
            print("Epoch {}".format(epoch+1))
            resume_path = os.path.join(args.folder, "%03d.pth" % epoch)
            print('=> loading pth from {} ...'.format(resume_path))
            utils.load_dict(resume_path, model)
            loss = do_evaluate(val_loader, model, criterion, args)
            print("Evaluate loss : {:.4f}".format(loss))

            losses.append(loss)
            if best_loss:
                if loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch + 1
            else:
                best_loss = loss
        print()
        print("Best loss : {:.4f} Appears in {}".format(best_loss, best_epoch))

        max_loss = max(losses)

        plt.figure(figsize=(7,7))

        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.xlim((0,args.epochs+1)) 
        plt.ylim([0, float('%.1g' % (1.22*max_loss))])

        plt.scatter(range(1, args.epochs+1), losses, s=9)

        plt.savefig("figs/evalall.jpg")

    else:
        print('=> loading pth from {} ...'.format(args.resume))
        utils.load_dict(args.resume, model)
        loss = do_evaluate(val_loader, model, criterion, args)
        print("Evaluate loss : {:.4f}".format(loss))


def do_evaluate(val_loader, model, criterion, args):
    batch_time = utils.AverageMeter('Time', ':6.2f')
    data_time = utils.AverageMeter('Data', ':2.2f')
    losses = utils.AverageMeter('Loss', ':.4f')
    
    progress = utils.ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses],
        prefix="Evaluate ")
    end = time.time()

    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(input)

            loss = criterion(output, target)

            # record loss
            losses.update(loss.item(), input.size(0))          
            batch_time.update(time.time() - end)        
            end = time.time()   

            if i % args.print_freq == 0:
                progress.display(i)
    
    return losses.avg

if __name__ == '__main__':

    args = get_args()

    main(args)


