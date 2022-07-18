"""
    Provide functionality to map between point clouds and images.
"""
import torch


def img2pc(feature_img, points_2d):
    # Bilinear interpolation to get point feature from the Image feature map
    # feature_img: (B, C_img, H, W); points_2d: (B, N, 2)
    B, C_img, H, W = feature_img.size()
    N = points_2d.size(1)
    batch_idx = torch.arange(B).unsqueeze(-1).repeat(1, N).flatten()    # (B*N)
    points_2d = points_2d.view(-1, 2)                                   # (B*N)
    x, y = points_2d[:, 0], points_2d[:, 1]
    # query corresponding img feature with the 2D coordinates
    new_pc_features = torch.zeros(B*N, C_img, device=feature_img.device)
    total_weight = 0
    for ty in [torch.ceil(y), torch.floor(y)]:
        for tx in [torch.ceil(x), torch.floor(x)]:
            weight = (1-torch.abs(x-tx)) * (1-torch.abs(y-ty))
            temp_feature_img = feature_img[batch_idx, :, ty.long().clamp(max=H-1), tx.long().clamp(max=W-1)]
            new_pc_features = new_pc_features + weight.unsqueeze(1) * temp_feature_img
            total_weight = total_weight + weight.unsqueeze(1)
    new_pc_features = new_pc_features / total_weight
    new_pc_features = new_pc_features.view(B, N, C_img).transpose(-1, -2)
    return new_pc_features  # (B, C_img, N)

def pc2img(feature_img, feature_pc, points_2d, idx_map):
    # feature_img: (B, C_img, H, W); feature_pc: (B, C_pc, N); points_2d: (B, N, 2); idx_map: (B, 2, H, W)
    B, C_img, H, W = feature_img.size()
    C_pc = feature_pc.size(1)
    # scatter point features to a new feature map
    new_img_features = torch.zeros(B, C_pc, H, W, device=feature_img.device)
    batch_idx, point_idx = idx_map.split(1, dim=1)
    batch_idx, point_idx = batch_idx.view(-1), point_idx.view(-1)
    valid = point_idx!=-1
    batch_idx, point_idx = batch_idx[valid], point_idx[valid]
    x, y = points_2d[batch_idx, point_idx, 0], points_2d[batch_idx, point_idx, 1]
    temp_feature_pc = feature_pc[batch_idx, :, point_idx]
    x, y = torch.floor(x).long(), torch.floor(y).long()
    new_img_features[batch_idx, :, y, x] = temp_feature_pc
    return new_img_features # (B, C_pc, H, W)

def build_img2pc_map(feature_img, points_3d, points_2d, unique2d_idx):
    # indicating which point index shall the image pixel look for
    # feature_img: (B, C, H, W); points_3d: (B, 3, N); points_2d:(B, N, 2); unique2d_idx: (N1+N2+..., 2)
    distances = torch.norm(points_3d, dim=1, keepdim=True)
    batch_idx, point_idx = unique2d_idx[:, 0], unique2d_idx[:, 1]
    x, y = points_2d[batch_idx, point_idx, 0], points_2d[batch_idx, point_idx, 1]
    B, C, H, W = feature_img.size()
    distance_map = torch.ones(B, 1, H, W, device=feature_img.device) * (-9999)
    x, y = torch.floor(x).long(), torch.floor(y).long()
    distance_map[batch_idx, :, y, x] = -distances[batch_idx, :, point_idx]   # save negative distance, to suit max_pooling
    idx_map = torch.ones(B, 2, H, W, device=feature_img.device, dtype=torch.int64) * (-1)  # -1 for padding, no points mapped to the pixel
    idx_map[batch_idx, :, y, x] = unique2d_idx
    return distance_map, idx_map

def handel_pooling(distance_map, idx_map, points_2d, out_shape):
    # distance_map: (B, 1, H, W); idx_map: (B, 2, H, W); points_2d: (B, N, 2)
    B, _, H, W = distance_map.size()
    newH, newW = out_shape
    distance_map, indices = torch.nn.AdaptiveMaxPool2d(out_shape, return_indices=True)(distance_map)
    # indices: (B, 1, H', W')
    batch_idx = torch.arange(B).view(-1, 1).repeat(1, newH*newW).view(-1)
    new_idx_map = idx_map.view(B, 2, -1)
    new_idx_map = new_idx_map[batch_idx, :, indices.view(-1)]   # (B*H'*W', 2)
    new_idx_map = new_idx_map.view(B, newH, newW, 2).permute(0, 3, 1, 2)
    # rescale points_2d
    x, y = points_2d.split(1, dim=-1)
    x = x * (newW/W)
    y = y * (newH/H)
    points_2d = torch.cat([x, y], dim=-1)
    return distance_map, new_idx_map, points_2d
