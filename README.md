# MTrans
The PyTorch implementation of [Multimodal Transformer for Automatic 3D Annotation and Object Detection](https://arxiv.org/abs/2207.09805), which has been accepted by ECCV2023.

## Installation
The code has been tested on PyTorch v1.9.1. 

IoU loss is required for training. Before running, please install the IoU loss package following this [doc](https://github.com/Cliu2/MTrans/tree/main/loss#readme).

## Data Preparation
The KITTI 3D detection dataset can be downloaded from the official webstie: [link](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

## Train
To train a MTrans with the KITTI dataset. Simply run:
> python train.py --cfg_file configs/MTrans_kitti.yaml

## Trained Model
Trained checkpoint can be downloaded from [here](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/lcon7_connect_hku_hk/EYyRIedDNolHr3GSq0b3CZoBgsyI3XVjjtz4STD97WtKUQ?e=IhhNk4).
Although we try to fix the random seeds, due to the randomness in some asynchronuous CUDA opearations and data preprocessing (e.g., point sampling), the result might not be exactly the same from run to run.

## References
The IoU loss module is borrowed from "https://github.com/lilanxiao/Rotated_IoU". We thank the author for providing a neat implementation of the IoU loss.
