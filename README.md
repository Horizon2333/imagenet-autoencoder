# Auto-Encoder trained on ImageNet

Train VGG-like and ResNet-like auto-encoder on image dataset like ImageNet


![imagenet-autoencoder/reconstruction.jpg at main · Horizon2333/imagenet-autoencoder (github.com)](https://github.com/Horizon2333/imagenet-autoencoder/blob/main/figs/reconstruction.jpg)


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
bash run/eval.sh vgg16 results/caltech256-vgg16/099.pth caltech101
```

For evaluating all checkpoints under specific folder:

```shell
bash run/evalall.sh {model architecture} {checkpoints path} {dataset name}
# For example
bash run/evalall.sh vgg16 results/caltech256-vgg16/ caltech101
```
When all checkpoints are evaluated, a scatter diagram ```figs/evalall.jpg``` will be generated to show the evaluate loss trend.

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
| :--------: | :---: | :---: | :---: | :---: | :------: | :------: | :------: | :-------: | :--------: |
| Caltech256 |   [link](https://drive.google.com/file/d/1gebnzAnFDpT9mmzr2dDVZ39FxqZHSuD4/view?usp=sharing)    |   [link](https://drive.google.com/file/d/1JRooEtKw2-2R_u-pswX2C8mAl_GgAlhH/view?usp=sharing)    |   [link](https://drive.google.com/file/d/12ysuL1rzIedcL_KD3VNDcZn9lGwxCWFu/view?usp=sharing)    |   [link](https://drive.google.com/file/d/1ydCY3llYJLL3asZ45-EGPUYxB-jlLVFo/view?usp=sharing)    |     [link](https://drive.google.com/file/d/1vokB8J17t34qk8qN37cVrEes06wzJzzG/view?usp=sharing)     |      [link](https://drive.google.com/file/d/1EMfNI6uAMdx-T1QmYg-UQHNWLxkaub6c/view?usp=sharing)    |     [link](https://drive.google.com/file/d/1-lA1dtP9q9ABom7c3qbMy7JYnnQvsI9H/view?usp=sharing)     |     [link](https://drive.google.com/file/d/1yNzkPhf2LAzu0mVm3ZedTObl_s-2pg1J/view?usp=sharing)      |     [link](https://drive.google.com/file/d/1HX7zaMK4ug6GjdUljG8Jqc4OvT8aLTrD/view?usp=sharing)      |
|  Objects365  |       |       |   [link](https://drive.google.com/file/d/16ozLClq8_Kpoc1Ln8dgIkQC4v7a1OTyz/view?usp=sharing)    |   [link](https://drive.google.com/file/d/1nR_9_WsYXGzBvzLdsxlEba9XyBwg1aD7/view?usp=sharing)    |          |          |     [link](https://drive.google.com/file/d/1FLPcRcAKaYBZrJQ7uYz0ST0WPrgacwm6/view?usp=sharing)     |     [link](https://drive.google.com/file/d/1pVtZpQn2kT1e2ZhG1MBvLLAMEVI30mVL/view?usp=sharing)      |           |
|  ImageNet  |       |       |   [link](https://drive.google.com/file/d/1WwJiQ1kBcNCZ37F6PJ_0bIL0ZeU3_sV8/view?usp=sharing)    |       |          |          |          |           |           |

Note that the size of Objects365 dataset is about half of ImageNet dataset(128 million images, much larger than Caltech256), so the performance may be comparable.
