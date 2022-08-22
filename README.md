# MTrans
The PyTorch implementation of 'Multimodal Transformer for Automatic 3D Annotation and Object Detection', which has been accepted by ECCV2022.

## Data Preparation
The KITTI 3D detection dataset can be downloaded from the official webstie: [link](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

## Train
To train a MTrans with the KITTI dataset. Simply run:
> python train.py --cfg_file configs/MTrans_kitti.yaml

## Pretrained Model
COMING SOON.

## References
The IoU loss module is borrowed from "https://github.com/lilanxiao/Rotated_IoU". We thank the author for providing a neat implementation of the IoU loss.
