"""
    Contains other functionalities of manipulating point clouds.
"""

import numpy as np
import torch
import scipy.ndimage

def check_points_in_box(points3d, location, dimension, yaw):
    # points3d: (N, 3); box3d: [x, y, z, l, w, h, yaw]
    cx, cy, cz = location
    l, w, h = dimension
    center = np.array([cx, cy, cz])[None, ...] # (1, 3)
    rot = np.array([
        [np.cos(yaw), np.sin(yaw), 0],
        [-np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    points3d = points3d - center    # translation compensation
    points3d = np.matmul(points3d, rot.T)   # rotation compensation
    points3d = np.abs(points3d)
    x, y, z = points3d[:, 0], points3d[:, 1], points3d[:, 2]
    # by observation about the Kitti Dataset, some foregrounds are close to box but not exactly in it, so smooth the boundary a bit
    mask = np.logical_and.reduce((x<=l/2+0.1, y<=w/2+0.1, z<=h/2+0.04))   
    return mask

def build_rotation_matrix(yaw):
    # yaw: (B, 1)
    B = yaw.size(0)
    rot_matrix = torch.cat([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw), 
                            -torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw),
                            torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)], dim=-1)
    rot_matrix = rot_matrix.view(B, 1, 3, 3)
    return rot_matrix

def check_points_in_box_cuda(points3d, location, dimension, yaw):
    # points3d: (B, N, 3); location: (B, 3), dimension: (B, 3), yaw: (B, 1)
    B, N, _ = points3d.shape
    rot_matrix = build_rotation_matrix(yaw)
    points3d = points3d - location.unsqueeze(1)
    points3d = torch.matmul(points3d.unsqueeze(-2), rot_matrix.transpose(-1, -2))   # (B, N, 1, 3)
    points3d = torch.abs(points3d.squeeze(-2))                                      # (B, N, 3)
    tolerance = torch.tensor([0.05, 0.05, 0.01], device=points3d.device).view(1, 1, 3)
    inbox = torch.all(points3d < (dimension.unsqueeze(1)/2+tolerance).repeat(1, N, 1), dim=-1)
    return inbox    # (B, N)


def farthest_point_sample_2d(xy, npoint):
    """
    Input:
        xy: pointcloud data, [B, N, 2]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    Ref: [https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py]
    """
    device = xy.device
    B, N, C = xy.shape
    centroids = torch.zeros(B, npoint+1, dtype=torch.long, device=device)
    distance = torch.ones(B, N).to(device) * 1e10
    # farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    # take the last point as begining, which is the background (-1,-1), always remove this point
    farthest = -1    
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint+1):
        centroids[:, i] = farthest
        centroid = xy[batch_indices, farthest, :].view(B, 1, 2)
        dist = torch.sum((xy - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids[:, 1:]

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    Ref: [https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def random_sample_in_box(boxes, nums_to_sample, calibs):
    # boxes: (B, 7); nums_to_sample: (B); calibs: List
    points = []
    device = boxes.device
    boxes = boxes.detach().cpu()
    for i, box in enumerate(boxes):
        if nums_to_sample[i]>0:
            location, dimension, yaw = box[:3], box[3:6], box[6]
            p = torch.rand((nums_to_sample[i], 3))
            p = (p * 2 - 1) * dimension.abs() # range: [-l, -w, -h] ~ [l, w, h]
            rot = torch.tensor([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ], dtype=torch.float32)
            p = torch.matmul(p, rot.T)
            p = p + location
            p, _ = calibs[i].velo_to_cam(p)     # (n, 2)
            p = torch.from_numpy(p).to(device).float()
            points.append(p)    
    points = torch.cat(points, dim=0)
    return points

def build_image_location_map(B, H, W, device='cuda'):
    locations = torch.zeros(B, H, W, 3, dtype=torch.float32, device=device)
    batch_idx = torch.arange(B, dtype=torch.long, device=device)
    row_idx = torch.arange(H, dtype=torch.long, device=device)
    col_idx = torch.arange(W, dtype=torch.long, device=device)
    locations[batch_idx, : , :, 0] = batch_idx.view(-1, 1, 1).float()
    locations[:, row_idx, :, 1] = row_idx.view(1, -1, 1).float()
    locations[:, :, col_idx, 2] = col_idx.view(1, 1, -1).float()
    return locations

def random_sample_pixels(fore_logits, nums_to_sample, image):
    # fore_logits: B, 2, H, W
    # nums_to_sample: (B,)
    # image_mask: B, H, W
    device = fore_logits.device
    fore_mask = fore_logits.argmax(dim=1)==1  # B, H, W
    img_mask = (image!=0).all(dim=1)    # (B, H, W), True if not padding
    fore_mask = fore_mask * img_mask
    B, H, W = fore_mask.shape
    coords = build_image_location_map(B, H, W, device)      # B, H, W, 3
    masked_coords = coords[fore_mask].view(-1, 3)           # (f1+f2+...+fb, 3)

    all_points = []
    for i in range(B):
        idx = masked_coords[:, 0] == i
        candidates = masked_coords[idx]
        if candidates.size(0) > nums_to_sample[i]:
            idx = np.random.choice(torch.arange(candidates.size(0)), nums_to_sample[i].item(), replace=False)
            idx = torch.from_numpy(idx).to(device)
            points = candidates[idx]
        elif candidates.size(0) < nums_to_sample[i]:
            _, idx = fore_logits.softmax(dim=1)[i, 1, :, :].view(-1).topk(nums_to_sample[i])
            points = coords[i, :, :, :].view(3, -1)[:, idx].transpose(-1, -2)
        else:
            points = candidates
        all_points.append(points)
    all_points = torch.cat(all_points, dim=0)
    return all_points

def random_sample_masked_points(masked_points, nums_to_sample):
    # masked_points: (m1+m2+...+mb, 1+4+2); nums_to_sample: (B,)
    device = masked_points.device
    B = nums_to_sample.size(0)
    sample_idx = torch.arange(masked_points.size(0))
    batch_idx = masked_points[:, 0]
    all_idx = []
    for i in range(B):
        candidates = sample_idx[batch_idx == i]
        idx = np.random.choice(candidates, nums_to_sample[i].item(), replace=False)
        idx = torch.from_numpy(idx).to(device)
        all_idx.append(idx)

    all_idx = torch.cat(all_idx, dim=0)
    sampled_masked_points = masked_points[all_idx, :]
    return sampled_masked_points

def generate_random_mask(B, N, num_slots, nums_to_mask, device='cpu', training=True):
    # num_slots: (B, ) indicating the range that can be masked
    # nums_to_mask: (B, ), indicating numbers to mask for each row
    mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    if not training:
        # fix generated mask position for evaluation, so that every eval the same positions will be masked
        fix_gen = torch.Generator()
        fix_gen.manual_seed(314159)
        
    for i in range(B):
        n = nums_to_mask[i]
        if not training:
            idx = torch.randperm(num_slots[i].item(), generator=fix_gen)[:n]
        else:
            idx = torch.randperm(num_slots[i].item())[:n]
        mask[i, idx] = True
    return mask # (B, N)

def gaussian_sampling(n=112, sigma=55):
    # generate a 2D gaussian pdf with nxn shape
    map = np.zeros((n, n))
    map[n//2, n//2] = 1
    map = scipy.ndimage.gaussian_filter(map, sigma)
    return map

def build_image_location_map_single(H, W, device='cuda'):
    locations = torch.zeros(2, H, W, dtype=torch.float32, device=device)
    row_idx = torch.arange(H, dtype=torch.long, device=device)
    col_idx = torch.arange(W, dtype=torch.long, device=device)
    locations[1, row_idx, :] = row_idx.view(1, -1, 1).float()
    locations[0, :, col_idx] = col_idx.view(1, 1, -1).float()
    locations = locations.permute(1, 2, 0)  # H, W, 2 for (x, y) 2D coords
    return locations

def recover3d_by_2d_depth(coords_2d, depth, calibs):
    # coords_2d: (B, N, 2), depth: (B, N), calib: <list> (N,)
    B, _, N = coords_2d.shape
    coords_3d = []
    for i in range(B):
        coords_3d = calibs[i].img_to_velo(coords_2d[i], depth[i])
    coords_3d = torch.stack(coords_3d, dim=0)
    return coords_3d

def knn(coords_2d, context_2d, k):
    # coords_2d: (B, N, 2), context_2d(B, M, 2), k:int
    dist = torch.norm(coords_2d.unsqueeze(2) - context_2d.unsqueeze(1), dim=-1)  # (B, N, M)
    ktop = torch.topk(-dist, k, dim=-1).indices
    return ktop     # (B, N, k)

def get_knn_values(idx, features):
    # features: (B, M, f), idx: (B, N, k)
    B, M, f = features.shape
    _, N, k = idx.shape
    bid = torch.arange(B, device=idx.device).view(-1, 1, 1)*M
    idx = idx + bid
    features = features.view(B*M, f)[idx.view(-1), :]
    features = features.view(B, N, k, f)
    return features

