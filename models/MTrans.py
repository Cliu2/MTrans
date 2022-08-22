import torch
import torchvision
from torch import nn
from models.CNN_backbone.FPN import FPN
from models.modules.modality_mapper import img2pc
from utils.point_ops import build_image_location_map_single
from models.modules.point_sa import AttentionPointEncoder
from loss import cal_diou_3d
import numpy as np
import math

class MTrans(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
        self.parameters_loaded = []     # record the names of parameters loaded from previous stage

        # CNN image encoder
        cimg, cpts = cfgs.POINT_ATTENTION.input_img_channel, cfgs.POINT_ATTENTION.input_pts_channel
        self.cnn = FPN(4, 128, cimg)

        # MAttn Transformer
        self.attention_layers = AttentionPointEncoder(cfgs.POINT_ATTENTION)
        self.xyzd_embedding = nn.Sequential(
            nn.Linear(3, cpts),
            nn.LayerNorm(cpts),
            nn.ReLU(inplace=True),
            nn.Linear(cpts, cpts)
        )
        self.unknown_f3d = nn.Parameter(torch.zeros(cpts))
        self.unknown_f3d = nn.init.normal_(self.unknown_f3d)
        
        hidden_size = cfgs.POINT_ATTENTION.hidden_size
        # heads: 3D box regression
        self.foreground_head = nn.Sequential(
            nn.Linear(hidden_size+cimg+3, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 2)
        )
        self.xyz_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.4),
            nn.Linear(512, 3)
        )
        
        if cfgs.POINT_ATTENTION.use_cls_token:
            self.box_head = nn.Sequential(
                nn.Linear(hidden_size+cimg+hidden_size, 512),
                nn.LayerNorm(512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=cfgs.box_drop),
                nn.Linear(512, 7)
            )
            self.conf_dir_head = nn.Sequential(
                nn.Linear(hidden_size+cimg+hidden_size, 512),
                nn.LayerNorm(512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.4),
                nn.Linear(512, 3)
            )
        else:
            raise RuntimeError

        # Image transformations, apply data augmentation dynamically rather than in dataloader
        self.pred_transforms = torch.nn.Sequential(
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )
        self.train_transforms = torch.nn.Sequential(
            torchvision.transforms.RandomAutocontrast(p=0.5),
            torchvision.transforms.RandomAdjustSharpness(np.random.rand()*2, p=0.5),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.5, 0.5, 0.5, 0.3)], p=0.5),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

    def build_target_mask(self, B, N, starts, ends, device='cpu'):
        # starts: (B, ); ends: (B, )
        # fill each row with 'True', starting from the start_idx, to the end_idx
        mask = [([0]*starts[i] + [1] * (ends[i]-starts[i]) + [0] * (N-ends[i])) for i in range(B)]
        mask = torch.tensor(mask, dtype=torch.bool, device=device)
        return mask     # (B, N)

    def forward(self, data_dict):
        image = data_dict.images                                        # (B, C, H, W)
        overlap_masks = data_dict.overlap_masks                         # (B, 1, H, W)
        sub_cloud = data_dict.sub_clouds[:, :, :3]                      # (B, N, 3) for x, y, z point cloud
        sub_cloud2d = data_dict.sub_clouds2d                            # (B, N, 2) for projected point cloud
        ori_cloud2d = data_dict.ori_clouds2d                            # (B, N, 2) for original 2D coords refer to the full image
        real_point_mask = data_dict.real_point_mask                     # (B, N), 1-realpoint; 0-padding; 2-mask; 3-jitter
        foreground_label = data_dict.foreground_label                   # (B, N), 1-foreground; 0-background; 2-unknown
        device = sub_cloud.device
        pred_dict = {
            'batch_size': image.size(0)
        }

        impact_points_mask = real_point_mask==1 # (B, N), points with known 3D coords, unmasked, unjittered
        unmaksed_known_points = (impact_points_mask) + (real_point_mask==3) # (B, N), points has 3D coords, no masked, no padding
        nonpadding_points = (unmaksed_known_points) + (real_point_mask==2)  # (B, N), points known, no padding

        # normalize point cloud
        sub_cloud_center = (sub_cloud * impact_points_mask.unsqueeze(-1)).sum(dim=1) / (impact_points_mask.sum(dim=1, keepdim=True)+1e-6)  # (B, 3)
        # only norm x&y coords
        # sub_cloud_center[:,-1] = 0
        data_dict.locations = data_dict.locations - sub_cloud_center
        sub_cloud = sub_cloud - sub_cloud_center.unsqueeze(1)
        data_dict.sub_clouds[:, :, :3] = data_dict.sub_clouds[:, :, :3] - sub_cloud_center.unsqueeze(1)
        sub_cloud = sub_cloud * (nonpadding_points).unsqueeze(-1)
        pred_dict['subcloud_center'] = sub_cloud_center
        # random augment point cloud with 50% probability
        if self.training:
            random_shift = (torch.rand(sub_cloud.size(0), 1, 3, device=sub_cloud.device) * torch.tensor([[[2, 2, 0.5]]], device=sub_cloud.device) \
                            - torch.tensor([[[1, 1, 0.25]]], device=sub_cloud.device))
            random_shift = random_shift * torch.randint(0, 2, (sub_cloud.size(0), 1, 1), device=random_shift.device).bool()
            sub_cloud = sub_cloud + random_shift
            data_dict.locations = data_dict.locations + random_shift.squeeze(1)
        # random scale point cloud
        if self.training:
            random_scale = (torch.rand(sub_cloud.size(0), 1, 3, device=sub_cloud.device) * 0.2 + 0.9)
            random_idx = torch.randint(0, 2, (sub_cloud.size(0), 1, 1), device=random_scale.device).bool()
            random_scale = random_scale * random_idx + 1 * (~random_idx)

            sub_cloud = sub_cloud * random_scale
            data_dict.locations = data_dict.locations * random_scale.squeeze(1)
            data_dict.dimensions = data_dict.dimensions * random_scale.squeeze(1)
        # random flip
        if self.training:
            # flip y coords
            random_idx = torch.randint(0, 2, (sub_cloud.size(0), 1), device=sub_cloud.device) * 2 - 1  # [-1, 1]
            sub_cloud[:, :, 1] = sub_cloud[:, :, 1] * random_idx
            data_dict.locations[:, 1] = data_dict.locations[:, 1] * random_idx.squeeze(1)
            data_dict.yaws = data_dict.yaws * random_idx
            data_dict.sub_clouds = sub_cloud

            flipped_image = torch.flip(image, dims=(-1,))
            image = image * (random_idx!=-1).view(-1, 1, 1, 1) + flipped_image * (random_idx==-1).view(-1, 1, 1, 1)
            flipped_x = image.size(-1) - 1 - sub_cloud2d[:, :, 0]
            sub_cloud2d[:,:,0] = sub_cloud2d[:,:,0] * (random_idx!=-1) + flipped_x * (random_idx==-1)
            data_dict.images = image
            data_dict.sub_cloud2d = sub_cloud2d

        pred_dict['gt_coords_3d'] = sub_cloud

        # random augment input image
        if self.training:
            image = torch.cat([self.train_transforms(image), overlap_masks], dim=1)
        else:
            # normalize image
            image = torch.cat([self.pred_transforms(image), overlap_masks], dim=1)

        # 1. extract information of images 
        B, _, H, W = image.size()
        image_features, _ = self.cnn(image)
        
        # 2. build new cloud, which contains blank slots to be interpolated
        B, N, _ = sub_cloud.size()
        scale = self.cfgs.sparse_query_rate
        qH, qW = H//scale, W//scale
        # hide and jitter point cloud
        jittered_cloud = sub_cloud.clone()
        jittered_cloud[(real_point_mask==3).unsqueeze(-1).repeat(1, 1, 3)] += torch.rand((real_point_mask==3).sum()*3, device=sub_cloud.device)*0.1 - 0.05
        jittered_cloud = jittered_cloud * (unmaksed_known_points.unsqueeze(-1))

        key_c2d = sub_cloud2d                                                                   # (B, N, 2)
        key_f2d = img2pc(image_features, key_c2d).transpose(-1, -2)                             # (B, N, Ci)
        key_f3d = self.xyzd_embedding(jittered_cloud) * (unmaksed_known_points.unsqueeze(-1)) + \
                  self.unknown_f3d.view(1, 1, -1).repeat(B, N, 1) * (~unmaksed_known_points.unsqueeze(-1))
        query_c2d = (build_image_location_map_single(qH, qW, device)*scale).view(1, -1, 2).repeat(B, 1, 1)     # (B, H*W, 2)
        query_f2d = img2pc(image_features, query_c2d).transpose(-1, -2)                         # (B, H*W, Ci)
        query_f3d = self.unknown_f3d.view(1, 1, -1).repeat(B, query_f2d.size(1), 1)


        # 3. Self-attention to decode missing 3D features
        # only unmasked known foreground will be attended
        attn_mask = unmaksed_known_points
        query_f3d, key_f3d, cls_f3d = self.attention_layers(query_c2d, query_f2d, query_f3d, key_c2d, key_f2d, key_f3d, attn_mask)
        pred_key_coords_3d = self.xyz_head(key_f3d)                                                # (B, N, 3)
        pred_dict['pred_coords_3d'] = pred_key_coords_3d
        
        diff_xyz = (pred_key_coords_3d.detach() - sub_cloud) * nonpadding_points.unsqueeze(-1)
        pred_key_foreground = self.foreground_head(torch.cat([key_f2d, key_f3d, diff_xyz], dim=-1))   # (B, N, 2)
        pred_dict['pred_foreground_logits'] = pred_key_foreground

        pred_query_coords_3d = self.xyz_head(query_f3d)
        pred_dict['enriched_points'] = pred_query_coords_3d
        pred_query_foreground = self.foreground_head(torch.cat([query_f2d, query_f3d, torch.zeros_like(pred_query_coords_3d)], dim=-1))
        pred_dict['enriched_foreground_logits'] = pred_query_foreground

        # 4. Predict 3D box
        # norm center
        all_points = torch.cat([sub_cloud], dim=1)
        all_forground_mask = torch.cat([pred_key_foreground], dim=1).argmax(dim=-1, keepdim=True)*unmaksed_known_points.unsqueeze(-1)
        seg_center = (all_points*all_forground_mask).sum(dim=1) / ((all_forground_mask).sum(dim=1)+1e-6)
        data_dict.locations = data_dict.locations - seg_center
        pred_dict['second_offset'] = seg_center

        # make the logits more discriminative
        query_fore_logits = (pred_query_foreground*5).softmax(dim=-1)
        key_fore_logits = (pred_key_foreground*5).softmax(dim=-1)
        global_f3d = ((query_f3d * query_fore_logits[:, :, 1:2]).sum(dim=1) + (key_f3d * key_fore_logits[:, :, 1:2]).sum(dim=1)) / \
                    (query_fore_logits[:, :, 1:2].sum(dim=1) + key_fore_logits[:, :, 1:2].sum(dim=1) + 1e-10)         # (B, Cp)
        global_f2d = torch.nn.AdaptiveAvgPool2d((1,1))(image_features).squeeze(-1).squeeze(-1)  # (B, Ci)

        box_feature = torch.cat([cls_f3d.squeeze(1), global_f3d, global_f2d], dim=-1)
        box = self.box_head(box_feature)
        conf_dir_pred = self.conf_dir_head(box_feature)

        location, dimension, yaw = box[:, 0:3], box[:, 3:6], box[:, 6:7]
        direction, confidence = conf_dir_pred[:, 0:2], conf_dir_pred[:, 2:3]
        confidence = torch.nn.Sigmoid()(confidence)

        dim_anchor = torch.tensor(self.cfgs.anchor, device=device).view(1, 3)
        da = torch.norm(dim_anchor[:, :2], dim=-1, keepdim=True)
        ha = dim_anchor[:, 2:3]
        pred_loc = location * torch.cat([da, da, ha], dim=-1)
        pred_dim = torch.exp(dimension) * dim_anchor
        pred_yaw = torch.tanh(yaw)
        pred_yaw = torch.arcsin(pred_yaw)

        pred_dict['location'] = pred_loc
        pred_dict['dimension'] = pred_dim
        pred_dict['yaw'] = pred_yaw
        pred_dict['direction'] = direction
        pred_dict['conf'] = confidence

        return pred_dict

    def get_loss(self, pred_dict, data_dict):
        loss_dict = {}
        B = pred_dict['batch_size']
        has_label = data_dict.use_3d_label                      # (B)
        real_point_mask = data_dict.real_point_mask            # (B, N)

        # 1. foreground loss
        segment_logits = pred_dict['pred_foreground_logits'].transpose(-1,-2)   # (B, 2, N)
        gt_segment_label = data_dict.foreground_label                     # (B, N)
        # loss only for those have 3D label
        segment_gt, segment_logits = gt_segment_label[has_label], segment_logits[has_label]   
        loss_segment = nn.CrossEntropyLoss(reduction='none', ignore_index=2)(segment_logits, segment_gt)
        # balance fore and background, take mean across batch samples
        lseg = 0
        if (segment_gt==1).sum() > 0:
            lseg = lseg + (loss_segment * (segment_gt==1)).sum(dim=-1) / ((segment_gt==1).sum(dim=-1)+1e-6)
        if (segment_gt==0).sum() > 0:
            lseg = lseg + (loss_segment * (segment_gt==0)).sum(dim=-1) / ((segment_gt==0).sum(dim=-1)+1e-6)
        loss_segment = lseg.mean()
        # dice loss
        segment_prob = segment_logits.softmax(dim=1)[:, 1, :]
        inter = 2 * (segment_prob * (segment_gt==1)).sum(dim=-1) + 1e-6
        uni = (segment_prob*(segment_gt!=2)).sum(dim=-1) +(segment_gt==1).sum(dim=-1) + 1e-6
        dice_loss = 1 - inter/uni
        loss_segment = loss_segment + dice_loss.mean()

        # metric: Segment IoU
        segment_pred = segment_logits.argmax(dim=1) * (segment_gt!=2)
        intersection = (segment_pred*(segment_gt==1)).sum(dim=1)
        union = ((segment_pred + (segment_gt==1)).bool()).sum(dim=1)+1e-10

        seg_iou = (intersection / union).mean()
        loss_dict['loss_segment'] = (loss_segment.item(), B, 'losses')
        loss_dict['segment_iou'] = (seg_iou.item(), B, 'segment_iou')

        # 2. depth loss
        gt_coords = pred_dict['gt_coords_3d']                     # (B, N, 3)
        pred_coords = pred_dict['pred_coords_3d']                 # (B, N, 3)
        loss_mask = pred_dict['pred_foreground_logits'].argmax(dim=-1).float()  # ignore background
        loss_mask = (loss_mask) * (real_point_mask!=0)                          # ignore padding

        loss_depth = nn.SmoothL1Loss(reduction='none')(pred_coords, gt_coords).sum(dim=-1)
        loss_depth = loss_depth * loss_mask

        # balance mask/jitter/impact points
        l = 0
        l = l + (loss_depth * (real_point_mask==1)).sum(dim=1) / (((real_point_mask==1)*loss_mask).sum(dim=1)+1e-6) * 0.1
        l = l + (loss_depth * (real_point_mask==2)).sum(dim=1) / (((real_point_mask==2)*loss_mask).sum(dim=1)+1e-6)
        assert (real_point_mask!=3).all()
        loss_depth = l.mean()

        # metric: mean xyz err
        err_dist = torch.norm(pred_coords - gt_coords, dim=-1)
        err_dist = (err_dist * (loss_mask==1) * (real_point_mask==2)).sum(dim=-1) / (((loss_mask==1) * (real_point_mask==2)).sum(dim=-1)+1e-6)
        loss_dict['loss_depth'] = (loss_depth.item(), B, 'losses')
        loss_dict['err_dist'] = (err_dist.mean().item(), B, 'err_dist')

        # 3. box loss
        pred_loc, pred_dim, pred_yaw = pred_dict['location'], pred_dict['dimension'], pred_dict['yaw']
        gt_loc, gt_dim, gt_yaw = data_dict.locations, data_dict.dimensions, data_dict.yaws
        pred_loc, pred_dim, pred_yaw = pred_loc[has_label], pred_dim[has_label], pred_yaw[has_label]
        gt_loc, gt_dim, gt_yaw = gt_loc[has_label], gt_dim[has_label], gt_yaw[has_label]
        gt_boxes = torch.cat([gt_loc, gt_dim, gt_yaw], dim=-1)
        
        num_gt_samples = has_label.sum()
        pred_boxes = torch.cat([pred_loc, pred_dim, pred_yaw], dim=-1)
        diff_yaw = torch.sin(pred_yaw-gt_yaw)
        l_iou, iou3d, iou2d = cal_diou_3d(pred_boxes.unsqueeze(1), gt_boxes.unsqueeze(1))
        loss_box = l_iou.mean()

        # loss for direction
        gt_dir = self.clamp_orientation_range(gt_yaw)
        gt_dir = ((gt_dir>= -np.pi/2) * (gt_dir< np.pi/2)).long().squeeze(-1)
        pred_dir = pred_dict['direction'][has_label]
        loss_dir = torch.nn.CrossEntropyLoss()(pred_dir, gt_dir)
        acc_dir = (pred_dir.argmax(dim=-1) == gt_dir).sum() / num_gt_samples
        
        # loss for confidence
        confidence = pred_dict['conf'][has_label]
        loss_conf = torch.nn.SmoothL1Loss()(confidence, iou3d)
        err_conf = (confidence - iou3d).abs().sum() / num_gt_samples
        assert not iou3d.isnan().any()
        
        loss_dict['loss_box'] = (loss_box.item(), num_gt_samples, 'losses')
        loss_dict['iou3d'] = (iou3d.mean().item(), num_gt_samples, 'iou')
        loss_dict['iou2d'] = (iou2d.mean().item(), num_gt_samples, 'iou')
        loss_dict['err_loc'] = ((pred_loc - gt_loc).norm(dim=-1).mean().item(), num_gt_samples, 'box_err')
        loss_dict['err_dim'] = ((pred_dim - gt_dim).abs().mean().item(), num_gt_samples, 'box_err')
        loss_dict['err_yaw'] = (diff_yaw.abs().mean().item(), num_gt_samples, 'box_err')
        loss_dict['recall_7'] = ((iou3d>0.7).float().mean().item(), num_gt_samples, 'recall')
        loss_dict['recall_5'] = ((iou3d>0.5).float().mean().item(), num_gt_samples, 'recall')
        loss_dict['recall_3'] = ((iou3d>0.3).float().mean().item(), num_gt_samples, 'recall') 
        loss_dict['err_conf'] = (err_conf.item(), num_gt_samples, 'box_acc')
        loss_dict['acc_dir'] = (acc_dir.item(), num_gt_samples, 'box_acc')
        iou3d_histo = iou3d.detach().cpu()

        loss = loss_segment + loss_depth + loss_box*5 + loss_conf + loss_dir
        loss_dict['loss'] = (loss.item(), B, 'loss')

        return loss_dict, loss, loss_segment, loss_depth, loss_box, iou3d_histo

    def clamp_orientation_range(self, angles):
        # angles: (B, 1)
        a = angles.clone()
        for i in range(a.size(0)):
            while a[i] > np.pi:
                a[i] = a[i] - np.pi * 2
            while a[i] <= -np.pi:
                a[i] = a[i] + np.pi*2
        assert (a<=np.pi).all() and (a>=-np.pi).all()
        return a

    def adjust_direction(self, yaw, dir):
        # yaw: (B, 1), dir: (B, 1) - long
        yaw = self.clamp_orientation_range(yaw)
        for i in range(yaw.size(0)):
            # check direction
            if dir[i]==1 and not (yaw[i]>=-np.pi/2 and yaw[i]<np.pi/2):
                    yaw[i] = yaw[i] + np.pi
            elif dir[i]==0 and (yaw[i]>=-np.pi/2 and yaw[i]<np.pi/2):
                    yaw[i] = yaw[i] + np.pi
        return yaw

    def format_kitti_labels(self, pred_dict, data_dict, with_score=True):
        location, dimension, yaw = pred_dict['location'], pred_dict['dimension'], pred_dict['yaw']
        location = location + pred_dict['subcloud_center'] + pred_dict['second_offset']
        direction = pred_dict['direction'].argmax(dim=-1)
        yaw = self.adjust_direction(yaw, direction)
        labels = []
        for i in range(pred_dict['batch_size']):
            c = data_dict.calibs[i]
            x, y, z = location[i]
            l, w, h = dimension[i]

            a = yaw[i]

            a = -(a + np.pi/2)
            while a > np.pi:
                a = a - np.pi * 2
            while a <= -np.pi:
                a = a + np.pi*2
            a = round(a.item(), 2)
            
            z = z - h/2
            loc = torch.stack([x, y, z], dim=-1)
            loc = c.lidar_to_rect(loc.detach().cpu().unsqueeze(0).numpy())[0]
            loc = loc.round(2)
            dim = torch.stack([h, w, l], dim=-1).detach().cpu().numpy()
            dim = dim.round(2)
            x, y, z = loc
            alpha = a + math.atan2(z,x)+1.5*math.pi
            if alpha > math.pi:
                alpha = alpha - math.pi * 2
            elif alpha <= -math.pi:
                alpha = alpha + math.pi*2
            box_2d = ' '.join([f'{x:.2f}' for x in data_dict.boxes_2d[i].detach().cpu().numpy()])
            dim = ' '.join([f'{x:.2f}' for x in dim])
            loc = ' '.join([f'{x:.2f}' for x in loc])
            truncated = data_dict.truncated[i]
            occluded = data_dict.occluded[i]
            score = pred_dict['conf'][i].item()

            if 'scores' in data_dict.keys():
                # for test result, MAPGen confidence * 2D Box score
                score = score * data_dict['scores'][i] / max(pred_dict['conf']).item()

            if with_score:
                labels.append(f'{data_dict.class_names[i]} {truncated:.2f} {occluded} {alpha:.2f} {box_2d} {dim} {loc} {a:.2f} {score:.4f}')
            else:
                labels.append(f'{data_dict.class_names[i]} {truncated:.2f} {occluded} {alpha:.2f} {box_2d} {dim} {loc} {a:.2f}')
        return labels, data_dict.frames
