"""
Data loader for Kitti-tracking dataset.
The module should merge multiple samples in a batch, where each data sample is an <dict>, (referring to `kitti_tracking.py`).
    - Images from different samples will be in tensors of shape B, C, H, W.
    - Points from different samples will be concatenated anlong axis-0, with shape (N1+N2+...+Nk)*(1+f),
      where the first column denotes the index of sample that the point belongs to.
"""

from copy import deepcopy
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
from easydict import EasyDict

def kitti_collate_wrapper(batch:list):
    # merge a list of data sample into a batch
    use_3d_label = [obj['use_3d_label'] for obj in batch]
    images = [obj['frame_img'] for obj in batch]
    sub_clouds = [obj['sub_cloud'] for obj in batch]
    sub_clouds2d = [obj['sub_cloud2d'] for obj in batch]
    ori_clouds2d = [obj['ori_cloud2d'] for obj in batch]
    real_point_mask = [obj['real_point_mask'] for obj in batch]
    foreground_label = [obj['foreground_label'] for obj in batch]
    classes = [obj['class_idx'] for obj in batch]
    boxes_2d = [obj['box_2d'] for obj in batch]
    locations = [obj['location'] for obj in batch]
    dimensions = [obj['dimensions'] for obj in batch]
    yaws = [obj['yaw'] for obj in batch]
    overlap_masks = [obj['overlap_mask'] for obj in batch]
    truncated = [obj['truncated'] for obj in batch]
    occluded = [obj['occluded'] for obj in batch]
    frames = [obj['frame'] for obj in batch]
    class_names = [obj['class'] for obj in batch]

    # build numpy arrays
    use_3d_label = np.stack(use_3d_label, axis=0)
    sub_clouds = np.stack(sub_clouds, axis=0)
    sub_clouds2d = np.stack(sub_clouds2d, axis=0)
    ori_clouds2d = np.stack(ori_clouds2d, axis=0)
    real_point_mask = np.stack(real_point_mask, axis=0)
    foreground_label = np.stack(foreground_label, axis=0)    
    classes = np.array(classes)
    boxes_2d = np.stack(boxes_2d, axis=0)
    locations = np.stack(locations, axis=0)
    dimensions = np.stack(dimensions, axis=0)
    yaws = np.stack(yaws, axis=0)

    # turn into tensors
    use_3d_label = torch.from_numpy(use_3d_label).bool()
    images = torch.cat(images, dim=0)  
    overlap_masks = torch.cat(overlap_masks, dim=0)                             
    sub_clouds = torch.from_numpy(sub_clouds).float()               
    sub_clouds2d = torch.from_numpy(sub_clouds2d).float()   
    ori_clouds2d = torch.from_numpy(ori_clouds2d).float()
    real_point_mask = torch.from_numpy(real_point_mask).long()          
    foreground_label = torch.from_numpy(foreground_label).long()    
    classes = torch.from_numpy(classes).long()                      
    boxes_2d = torch.from_numpy(boxes_2d).float()                   
    locations = torch.from_numpy(locations).float()                 
    dimensions = torch.from_numpy(dimensions).float()               
    yaws = torch.from_numpy(yaws).unsqueeze(-1).float()             
    # position_masks = torch.cat(position_masks, dim=0)

    data = EasyDict({
        'use_3d_label': use_3d_label,               # (B, )
        'images':images,                            # (B, C, H, W)
        'overlap_masks': overlap_masks,             # (B, 1, H, W)
        'sub_clouds':sub_clouds,                    # (B, N, 4)
        'sub_clouds2d':sub_clouds2d,                # (B, N, 2)
        'ori_clouds2d':ori_clouds2d,                # (B, N, 2)
        'real_point_mask': real_point_mask,         # (B, N)
        'foreground_label': foreground_label,       # (B, N)
        'classes': classes,                         # (B, )
        'boxes_2d':boxes_2d,                        # (B, 4)
        'locations':locations,                      # (B, 3)
        'dimensions':dimensions,                    # (B, 3)
        'yaws':yaws,                                # (B, 1)
        'frames':frames,                            # list<string>, (B, )
        'truncated':truncated,                      # list<float>, (B, )
        'occluded':occluded,                        # list<int>, (B, )
        'class_names':class_names,                  # list<string>, (B, )
    })

    if 'calib' in batch[0].keys():
        calibs = [obj['calib'] for obj in batch]
        data['calibs'] = calibs
    if 'score' in batch[0].keys():
        scores = [obj['score'] for obj in batch]
        data['scores'] = scores
    return data

def build_kitti_loader(dataset, loader_config):
    # fix generator for reproducibility
    g = torch.Generator()
    g.manual_seed(loader_config.random_seed)

    loader = DataLoader(dataset, 
                        batch_size=loader_config.batch_size,
                        shuffle=loader_config.shuffle,
                        num_workers=loader_config.num_workers,
                        collate_fn=kitti_collate_wrapper,
                        pin_memory=loader_config.pin_memory,
                        drop_last=loader_config.drop_last,
                        generator=g)
    return loader

def move_to_cuda(data_dict, device):
    for k in data_dict.keys():
        if isinstance(data_dict[k], torch.Tensor):
            data_dict[k] = data_dict[k].to(device)
    return data_dict

def merge_two_batch(data_dict1, data_dict2):
    assert set(data_dict1.keys()) == set(data_dict2.keys())
    data_dict1 = deepcopy(data_dict1)
    for k in data_dict1.keys():
        if isinstance(data_dict1[k], torch.Tensor):
            data_dict1[k] = torch.cat([data_dict1[k], data_dict2[k]], dim=0)
        elif isinstance(data_dict1[k], list):
            data_dict1[k] = data_dict1[k] + data_dict2[k]
        else:
            raise RuntimeError(f"Not supported datatype{type(data_dict1[k])}")
    return data_dict1