import torch.nn as nn
import torch.nn.parallel as parallel

from . import vgg, resnet

def BuildAutoEncoder(args):

    if args.arch in ["vgg11", "vgg13", "vgg16", "vgg19"]:
        configs = vgg.get_configs(args.arch)
        model = vgg.VGGAutoEncoder(configs)

    elif args.arch in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        configs, bottleneck = resnet.get_configs(args.arch)
        model = resnet.ResNetAutoEncoder(configs, bottleneck)
    
    else:
        return None
    
    if args.parallel == 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = parallel.DistributedDataParallel(
                        model.to(args.gpu),
                        device_ids=[args.gpu],
                        output_device=args.gpu
                    )   
    
    else:
        model = nn.DataParallel(model).cuda()

    return model