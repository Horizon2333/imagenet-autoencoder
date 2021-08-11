# Auto-Encoder trained on ImageNet

Train VGG-like and ResNet-like auto-encoder on image dataset like ImageNet, Caltech256

1. [Project Structure](#project-structure)
2. [Install](#install)
3. [Data preparing](#data-preparing)
4. [Train and Evaluate](#train-and-evaluate)
5. [Tools](#tools)
6. [Model zoo](#model-zoo)

## Project Structure

```
$imagenet-autoencoder
	|──figs                 # result images
		|── *.jpg
	|──models
		|──builder.py       # build autoencoder models
		|──resnet.py        # resnet-like autoencoder
		|──vgg.py           # vgg-like autoencoder
	|──run
		|──eval.sh          # command to evaluate single checkpoint
		|──evalall.sh       # command to evaluate all checkpoints in specific folder
		|──train.sh         # command to train auto-encoder
    |──tools                
        |──decode.py        # decode random latent code to images
        |──encode.py        # encode single image to latent code
        |──generate_list.py # generate image list for training
        |──reconstrust.py   # reconstruct the images to see difference
    |──dataloader.py        # dataset and dataloader
    |──eval.py              # evaluate checkpoints
    |──train.py             # train models
    |──utils.py             # other utility function
    |──requirements.txt
    |──README.md
```

## Install


1. Clone the project
```shell
git clone https://github.com/Horizon2333/imagenet-autoencoder
cd imagenet-autoencoder
```
2. Install dependencies
```shell
pip install -r requirements.txt
```

## Data Preparing

Your dataset should looks like:

```
$your_dataset_path
    |──class1
        |──xxxx.jpg
        |──...
    |──class2
        |──xxxx.jpg
        |──...
    |──...
    |──classN
        |──xxxx.jpg
        |──...
```

The you can use ```tools/generate_list.py``` to generate list of training samples. Here we do not use ```torchvision.datasets.ImageFolder``` because it is very slow when dataset is pretty large. You can run

```shell
python tools/generate_list.py --name {name your dataset such as caltech256} --path {path to your dataset}
```

Then two files will be generated under ```list``` folder, one  ```*_list.txt``` save every image path and its class(here no use); one ```*_name.txt``` save index of every class and its class name.

## Train and Evaluate

For training

```shell
bash run/train.sh {model architecture such as vgg16} {you dataset name}
# For example
bash run/train.sh vgg16 caltech256
```

For evaluating single checkpoint:

```shell
bash run/eval.sh {model architecture} {checkpoint path} {dataset name}
# For example
bash run/eval.sh vgg16 results/caltech256-vgg16/099.pth caltech256
```

For evaluating all checkpoints under specific folder:

```shell
bash run/evalall.sh {model # For example
bash run/eval.sh vgg16 results/caltech256-vgg16/099.pth caltech256} {checkpoints path} {dataset name}
# For example
bash run/evalall.sh vgg16 results/caltech256-vgg16/ caltech256
```

For model architecture, now we support ```vgg11,vgg13,vgg16,vgg19``` and ```resnet18, resnet34, resnet50, resnet101, resnet152```.

## Tools

We provide several tools to better visualize the auto-encoder results.

```reconstruct.py```

Reconstruct images from original one. This code will sample 64 of them and save the comparison results to ```figs/reconstruction.jpg```.

```shell
python tools/reconstruct.py --arch {model architecture} --resume {checkpoint path} --val_list {*_list.txt of your dataset}
# For example
python tools/reconstruct.py --arch vgg16 --resume results/caltech256-vgg16/099.pth --val_list caltech101_list.txt
```

```encode.py``` and ```decode.py```

Encode image to latent code or decode latent code to images.

```encode.py``` can transfer single image to latent code.

```shell
python tools/encode.py --arch {model architecture} --resume {checkpoint path} --img_path {image path}
# For example
python tools/encode.py --arch vgg16 --resume results/caltech256-vgg16/099.pth --img_path figs/reconstruction.jpg
```

```decode.py``` can transform 128 random latent code to images.

```shell
python tools/decode.py --arch {model architecture} --resume {checkpoint path} 
# For example
python tools/decode.py --arch vgg16 --resume results/caltech256-vgg16/099.pth
```

The decoded results will be save as ```figs/generation.jpg```

## Model zoo

|  Dataset   | VGG11 | VGG13 | VGG16 | VGG19 | ResNet18 | ResNet34 | ResNet50 | ResNet101 | ResNet152 |
| :--------: | :---: | :---: | :---: | :---: | :------: | :------: | :------: | :-------: | --------: |
| Caltech256 |       |       |       |       |          |          |          |           |           |
|  ImageNet  |       |       |       |       |          |          |          |           |           |

 Checkpoint will coming soon ...