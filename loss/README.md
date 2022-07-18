# Read me

## Compile the cuda extension library

```bash
# export PATH=/usr/local/$Version/bin/:$PATH
# export LD_LIBRARY_PATH=/usr/local/$Version/lib64:$LD_LIBRARY_PATH
nvcc -V

# To cuda extension library root folder
cd ./loss/cuda_op

# Run python to install
python setup.py install
```

## Using

>> enclosing type: aligned | pca | smallest(default)

* diou
```python
iou_loss, iou_3d, iou_2d = cal_diou_3d(pred, label, enclosing_type)
iou_loss = torch.mean(iou_loss)
iou_loss.backward()
```

* giou
```python
iou_loss, iou_3d, iou_2d = cal_giou_3d(pred, label, enclosing_type)
iou_loss = torch.mean(iou_loss)
iou_loss.backward()
```


## Test code

```python
import torch
import numpy as np
from loss import cal_diou_3d
from loss import cal_giou_3d

if __name__ == '__main__':
    box3d1 = np.array([1,1,1,2,2,2,np.pi/3])
    box3d2 = np.array([1,1,1,2,2,2,np.pi/3])
    tensor1 = torch.FloatTensor(box3d1).unsqueeze(0).unsqueeze(0).cuda()
    tensor2 = torch.FloatTensor(box3d2).unsqueeze(0).unsqueeze(0).cuda()

    loss, iou_3d, iou_2d = cal_giou_3d(tensor1, tensor1)
    print(loss)
    print(iou_3d)
    print(iou_2d)

    loss, iou_3d, iou_2d = cal_diou_3d(tensor1, tensor1)
    print(loss)
    print(iou_3d)
    print(iou_2d)
```
