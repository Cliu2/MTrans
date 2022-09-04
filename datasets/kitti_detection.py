"""
The customized Kitti Detection dataset for MAPGen. 
Each returned item will be an object with the features:
    1. object cropped image
    2. object frustum point cloud
    3. object 3D labels
    4. object frustum corresponding 2D coords
    5. foreground segmentation labels
"""

from utils.point_ops import check_points_in_box, build_image_location_map_single
from utils.calibrator import KittiCalibrator_detect
from utils.os_utils import verify_and_create_outdir
from torch.utils.data import Dataset
from easydict import EasyDict
from tqdm import tqdm
from PIL import Image
from os import path
import numpy as np
import torchvision
import pickle
import torch
import copy

class KittiDetectionDataset(Dataset):
    def __init__(self, data_root, cfg, **kwargs):
        super().__init__()
        
        self.cfg = cfg
        self.root = data_root
        self.classes = cfg.classes
        split = cfg.split
        if split in ['train', 'val']:
            split_folder = 'training'
        elif split == 'test':
            split_folder = 'testing'
        split_root = path.join(self.root, split_folder)
        self.split_root = split_root

        gt_set_path = path.join(data_root, 'gt_base', split, f'gt_set_{split}.pkl')
        if path.exists(gt_set_path):
            objects = pickle.load(open(gt_set_path, 'rb'))
        else:
            verify_and_create_outdir(path.join(data_root, 'gt_base', split))
            objects = self.build_dataset()
            pickle.dump(objects, open(gt_set_path, 'wb'))
    
        # filter objects by class and point cloud size
        self.objects = [o for o in objects if o['class'] in cfg.classes \
                            and o['sub_cloud'].shape[0] >= cfg.min_points \
                            and o['foreground_label'].sum() >= (5 if split!='test' else 0)]
        
        # use partial frames
        labeled_frames = np.unique([o['frame'] for o in objects])
        if 'partial_frames' in cfg.keys():
            self.labeled_frames = labeled_frames[:cfg.partial_frames]
            if cfg.get('use_3d_label', True):
                self.objects = [o for o in self.objects if o['frame'] in self.labeled_frames]
            else:
                self.objects = [o for o in self.objects if o['frame'] not in self.labeled_frames]

        # build guassian distribution for random sampling
        self.gaussian = torch.ones(cfg.out_img_size, cfg.out_img_size)
        self.gaussian = self.gaussian / self.gaussian.sum()
        self.img_coords = build_image_location_map_single(cfg.out_img_size, cfg.out_img_size, 'cpu')
        
    def __len__(self):
        return len(self.objects)

    def read_image(self, img_path):
        assert path.exists(img_path), f'{img_path} not exist'
        img = Image.open(img_path)
        img = torchvision.transforms.ToTensor()(img)    # (C, H, W)
        return img

    def read_point_cloud(self, pc_path):
        assert path.exists(pc_path), f'{pc_path} not exists'
        pc = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)  # (N, 4)
        return pc

    def read_calib(self, calib_path) -> KittiCalibrator_detect:
        return KittiCalibrator_detect(calib_path)

    def read_label(self, label_path, calib):
        with open(label_path) as f:
            labels = [l.strip().split(' ') for l in f.readlines()]
        object_labels = []
        for label in labels:
            if label[0] == 'DontCare':
                continue
            if label[0] not in self.classes:
                continue
            cls = label[0]
            truncated = float(label[1])
            if truncated > 0.95:    # remove too much truncated
                continue
            occluded = int(label[2])
            box_2d = np.array(label[4:8], dtype=np.float32)
            dim = np.array(label[8:11], dtype=np.float32)
            loc = np.array(label[11:14], dtype=np.float32)
            yaw = float(label[14])

            # change label coordinate system: camera sys -> lidar sys
            location = calib.rect_to_velo(loc[np.newaxis, ...])
            x, y, z = location[0]
            h, w, l = dim
            z += h/2
            yaw = -yaw - np.pi/2

            object_labels.append({
                'class': cls,
                'truncated': truncated,
                'occluded': occluded,
                'box_2d': box_2d,
                'dimensions': np.array([l, w, h]),
                'location': np.array([x, y, z]),
                'yaw': yaw,
            })

            if len(label)==16:
                score = float(label[15])
                object_labels[-1]['score'] = score
        return object_labels

    def read_test_rgb_detections(self, file):
        all_labels = {}
        with open(file) as f:
            lines = [l.strip() for l in f.readlines()]
            for l in lines:
                l = l.split(' ')
                cls = l[1]
                if cls not in self.classes:
                    continue
                frame = l[0]
                if frame not in all_labels.keys():
                    all_labels[frame] = []
                truncated=-1
                occluded=-1
                box_2d = np.array(l[3:7], dtype=np.float32)
                yaw = 0
                score = float(l[2])
                all_labels[frame].append({
                    'class': cls,
                    'truncated': truncated,
                    'occluded': occluded,
                    'box_2d': box_2d,
                    'dimensions': np.array([4,1.6,1.5]),
                    'location': np.array([0,0,0]),
                    'yaw': yaw,
                    'score': score
                })
        return all_labels

    def build_overlap_matrix(self, object_labels):
        num_labels = len(object_labels)
        overlap_matrix = [[] for _ in range(num_labels)]
        for i in range(num_labels):
            for j in range(i+1, num_labels):
                b1 = object_labels[i]['box_2d']     # left, top, right, bottom
                b2 = object_labels[j]['box_2d']
                overlap_vertical = max(b1[1], b2[1]) < min(b1[3], b2[3])
                overlap_horizontal = max(b1[0], b2[0]) < min(b1[2], b2[2]) 
                if overlap_vertical and overlap_horizontal:
                    overlap_matrix[i].append(j)
                    overlap_matrix[j].append(i)
        return overlap_matrix


    def build_dataset(self):
        print("========== Building Dataset ==========")
        all_objects = []
        split_root = self.split_root
        split = self.cfg.split
        out_3d_dir = path.join(self.root, 'processed', 'points_3d', split)
        out_2d_dir = path.join(self.root, 'processed', 'points_2d', split)
        try:
            verify_and_create_outdir(out_2d_dir)
            verify_and_create_outdir(out_3d_dir)
        except:
            print("[Warning] Preprocessed PointClouds and Images already exists.")

        with open(path.join(self.root, 'ImageSets', f'{split}.txt')) as f:
            all_frames = [l.strip() for l in f.readlines()]

        if self.cfg.split=='test':
            test_all_labels = self.read_test_rgb_detections(self.cfg.test_rgb_file)

        for frame in tqdm(all_frames, desc=f"Processing {split} data"):
            # preprocess frames, taking out the points within image scope, and their projected 2D coords
            img = self.read_image(path.join(split_root, 'image_2', f'{frame}.png'))
            H, W = img.shape[1:3]
            point_cloud = self.read_point_cloud(path.join(split_root, 'velodyne', f'{frame}.bin'))
            calib = self.read_calib(path.join(split_root, 'calib', f'{frame}.txt'))
            p2d_float, depth = calib.velo_to_cam(point_cloud[:, :3])
            x, y = p2d_float[:, 0], p2d_float[:, 1]
            idx = np.logical_and.reduce([depth>=0, x>=0, x<W, y>=0, y<H])
            point_cloud = point_cloud[idx]
            p2d_float = p2d_float[idx]
            point_cloud.astype(np.float32).tofile(path.join(out_3d_dir, f'{frame}.bin'))
            p2d_float.astype(np.float32).tofile(path.join(out_2d_dir, f'{frame}.bin'))

            ### build object-level dataset ###
            if self.cfg.split!='test':
                object_labels = self.read_label(path.join(split_root, 'label_2', f'{frame}.txt'), calib)
            else:
                if frame in test_all_labels.keys():
                    object_labels = test_all_labels[frame]
                else:
                    continue
            overlap_matrix = self.build_overlap_matrix(object_labels)
            for i, obj in enumerate(object_labels):
                obj['frame'] = frame
                # query sub cloud within the 2D box
                left, top, right, bottom = obj['box_2d']
                idx = np.logical_and.reduce([p2d_float[:, 0]>left, p2d_float[:, 1]>top, p2d_float[:, 0]<right, p2d_float[:, 1]<bottom])
                sub_cloud = point_cloud[idx]
                sub_cloud2d = p2d_float[idx]
                obj['sub_cloud'] = sub_cloud
                obj['sub_cloud2d'] = sub_cloud2d
                # generate foreground label
                foreground_label = check_points_in_box(sub_cloud[:, :3], location=obj['location'], dimension=obj['dimensions'], yaw=obj['yaw'])
                obj['foreground_label'] = foreground_label

                overlap_boxes = [object_labels[j]['box_2d'] for j in overlap_matrix[i]]
                obj['overlap_boxes'] = overlap_boxes

                all_objects.append(obj)
        
        return all_objects

    def __getitem__(self, index):
        obj = copy.deepcopy(self.objects[index])
        obj = self.load_object_full_data(obj)
        return obj

    def load_object_full_data(self, obj):
        class_name = obj['class']
        obj['class_idx'] = self.cfg.classes.index(class_name)
        obj['use_3d_label'] = self.cfg.get('use_3d_label', True)
        cloud_size = self.cfg.out_cloud_size

        split_root = self.split_root
        full_img = self.read_image(path.join(split_root, 'image_2', f"{obj['frame']}.png"))

        # build overlap mask
        overlap_mask = torch.ones_like(full_img[0:1, :, :])
        for olb in obj['overlap_boxes']:
            l, t, r, b = olb
            l, t, r, b = int(np.floor(l)), int(np.floor(t)), int(np.ceil(r)), int(np.ceil(b))
            overlap_mask[:, t:b, l:r] = 0
        full_img = torch.cat([full_img, overlap_mask], dim=0)

        l, t, r, b = obj['box_2d']
        # box2d augmentation, random scale + shift
        if self.cfg.get('box2d_augmentation', False):
            random_scale = np.random.rand(2) * 0.2 + 0.95        # [95% ~ 115%]
            random_shift = np.random.rand(2) * 0.1 - 0.05        # [-5% ~ 5%]
            tw, th, tx, ty = r-l, b-t, (r+l)/2, (t+b)/2
            tx, ty = tx + tw*random_shift[0], ty + th*random_shift[1]   # random shift
            tw, th = tw * random_scale[0], th*random_scale[1]           # random scale
            l, t, r, b = max(0, tx-tw/2), max(0, ty-th/2), min(tx+tw/2, full_img.shape[2]-1), min(ty+th/2, full_img.shape[1]-1)
        
            # re-crop frustum sub-cloud, and cloud's 2D projection
            frame = obj['frame']
            all_points = np.fromfile(path.join(self.root, 'processed', 'points_3d', self.cfg.split, f'{frame}.bin'), dtype=np.float32).reshape(-1,4)
            all_points2d = np.fromfile(path.join(self.root, 'processed', 'points_2d', self.cfg.split, f'{frame}.bin'), dtype=np.float32).reshape(-1,2)
            idx = np.logical_and.reduce([all_points2d[:, 0]>=l, all_points2d[:, 1]>=t, all_points2d[:, 0]<=r, all_points2d[:, 1]<=b])
            obj['sub_cloud'] = all_points[idx]
            obj['sub_cloud2d'] = all_points2d[idx]
            obj['foreground_label'] = check_points_in_box(obj['sub_cloud'][:, :3], location=obj['location'], dimension=obj['dimensions'], yaw=obj['yaw'])

        l, t, r, b = int(np.floor(l)), int(np.floor(t)), int(np.ceil(r)), int(np.ceil(b))
        img = full_img[:,t:b, l:r].unsqueeze(0)  # (1, 4, box_h, box_w)     

        # crop original image by the obj's 2D box
        box_size = max(b-t, r-l)
        out_shape = self.cfg.out_img_size
        img = torch.nn.functional.interpolate(img, scale_factor=out_shape/box_size, mode='bilinear', align_corners=True, recompute_scale_factor=False)
        h, w = img.shape[-2:]
        num_padding = (int(np.floor((out_shape-w)/2)), int(np.ceil((out_shape-w)/2)), int(np.floor((out_shape-h)/2)), int(np.ceil((out_shape-h)/2)))
        img = torch.nn.functional.pad(img, num_padding)     # zero-padding to make it square
        crop_sub_cloud2d = (obj['sub_cloud2d'] - np.array([l, t])) * (out_shape/box_size) + np.array([num_padding[0], num_padding[2]])
        assert np.logical_and.reduce([crop_sub_cloud2d[:, 0]>0, crop_sub_cloud2d[:,0]<out_shape, crop_sub_cloud2d[:, 1]>0, crop_sub_cloud2d[:,1]<out_shape]).all()            

        img, overlap_mask = img[:, 0:3, :, :], img[:, 3:4, :, :]

        # sampling the point cloud to fixed size
        out_sub_cloud = np.ones((cloud_size, 4))* (-9999)       # -9999 for paddings
        out_sub_cloud2d = np.ones((cloud_size, 2)) * (-9999)    # -9999 for paddings
        out_ori_cloud2d = np.ones((cloud_size, 2)) * (-9999)    
        out_real_point_mask = np.zeros((cloud_size))    # 0 for padding, 1 for real points, 2 for masked, 3 for jittered
        out_foreground_label = np.ones((cloud_size))*2   # 0 for background, 1 for foreground, 2 for unknown
        

        sub_cloud = obj['sub_cloud']
        sub_cloud2d = obj['sub_cloud2d']
        foreground_label = obj['foreground_label']
        out_cloud_size = self.cfg.out_cloud_size
        if sub_cloud.shape[0] > out_cloud_size:
            sample_idx = np.random.choice(np.arange(sub_cloud.shape[0]), out_cloud_size, replace=False)            # random sampling
            out_sub_cloud[...] = sub_cloud[sample_idx]
            out_sub_cloud2d[...] = crop_sub_cloud2d[sample_idx]
            out_ori_cloud2d[...] = sub_cloud2d[sample_idx]
            out_real_point_mask[...] = 1
            out_foreground_label[...] = foreground_label[sample_idx]    
        elif sub_cloud.shape[0] <= out_cloud_size:
            pc_size = sub_cloud.shape[0]
            out_sub_cloud[:pc_size] = sub_cloud
            out_sub_cloud2d[:pc_size] = crop_sub_cloud2d
            out_ori_cloud2d[:pc_size] = sub_cloud2d
            out_real_point_mask[:pc_size] = 1
            out_foreground_label[:pc_size] = foreground_label

            # sample 2D points, leave blank for 3D coords
            p = ((img[0]!=0).all(dim=0) * 1).numpy().astype(np.float64)    # only sample pixels from not-padding-area
            p = p / p.sum()
            resample = (p>0).sum() < (out_cloud_size - pc_size)
            sample_idx = np.random.choice(np.arange(out_shape * out_shape), out_cloud_size - pc_size, replace=resample,
                                          p=p.reshape(-1))
            sampled_c2d = self.img_coords.view(-1, 2)[sample_idx, :].numpy()
            out_sub_cloud2d[pc_size:, :] = sampled_c2d
            out_ori_cloud2d[pc_size:, :] = (sampled_c2d - np.array([num_padding[0], num_padding[2]])) / (out_shape/box_size) + np.array([l, t]) 
            

            assert np.logical_and.reduce([out_ori_cloud2d[:pc_size, 0]>=l, out_ori_cloud2d[:pc_size,0]<=r,
                                     out_ori_cloud2d[:pc_size, 1]>=t, out_ori_cloud2d[:pc_size,1]<=b]).all()

        # random mask/jitter points
        num_real_points = (out_real_point_mask==1).sum()
        mask_ratio = np.random.rand() * (self.cfg.mask_ratio[1] - self.cfg.mask_ratio[0]) + self.cfg.mask_ratio[0]     # randomly choose from (r_min, r_max)
        num_mask = min(int(mask_ratio * num_real_points), max(0, num_real_points - 5))        # leave at least 5 points
        idx = np.random.choice(np.arange(num_real_points), num_mask, replace=False)
        mask_idx = idx
        out_real_point_mask[mask_idx] = 2   # 2 for masked

        # load calib
        if self.cfg.load_calib:
            calib = self.read_calib(path.join(self.split_root, 'calib', f"{obj['frame']}.txt"))
            obj['calib'] = calib

        obj['frame_img'] = img
        obj['sub_cloud'] = out_sub_cloud
        obj['sub_cloud2d'] = out_sub_cloud2d
        obj['ori_cloud2d'] = out_ori_cloud2d
        obj['real_point_mask'] = out_real_point_mask
        obj['foreground_label'] = out_foreground_label
        obj['overlap_mask'] = overlap_mask

        return obj

def stat_dataset(dataset, info):
    print(f'\n\n####### Statistics {info} #######\n\n')
    for c in dataset.classes:
        class_objs = [o for o in dataset.objects if o['class']==c]
        num_samples = len(class_objs)
        print(f'==== {c} X {num_samples} ====')
        h = [b[3]-b[1] for b in [o['box_2d'] for o in class_objs]]
        w = [b[2]-b[0] for b in [o['box_2d'] for o in class_objs]]
        print('avg 2D (H, W): ', sum(h)/len(h), sum(w)/len(w))
        print('max 2D (H, W): ', max(h), max(w))
        print('min 2D (H, W): ', min(h), min(w))
        h = [b[0] for b in [o['dimensions'] for o in class_objs]]
        w = [b[1] for b in [o['dimensions'] for o in class_objs]]
        l = [b[2] for b in [o['dimensions'] for o in class_objs]]
        print('avg 3D (H, W, L): ', sum(h)/len(h), sum(w)/len(w), sum(l)/len(l))
        print('max 3D (H, W, L): ', max(h), max(w), max(l))
        print('min 3D (H, W, L): ', min(h), min(w), min(l))
        sc = [o['sub_cloud'].shape[0] for o in class_objs]
        print('avg points: ', sum(sc)/len(sc))
        print('max points: ', max(sc))
        print('min points: ', min(sc))
        foreground_points = [o['foreground_label'].sum() for o in class_objs]
        print('avg foreground points: ', sum(foreground_points)/len(foreground_points))
        print('max foreground points: ', max(foreground_points))
        print('min foreground points: ', min(foreground_points))
        print('\n\n')

if __name__=='__main__':
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description='Build Kitti Detection Dataset')
    parser.add_argument('-build_dataset', action='store_true')
    parser.add_argument('--cfg_file', type=str, help='Config file')
    args = parser.parse_args()
    cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))

    if args.build_dataset:
        root = cfg.data_root
        train_config = EasyDict(cfg.DATASET_CONFIG.TRAIN_SET)
        dataset = KittiDetectionDataset(root, train_config)
        val_config = EasyDict(cfg.DATASET_CONFIG.VAL_SET)
        dataset = KittiDetectionDataset(root, val_config)
    print('========== Finish Building ==========')
    

    dataset = KittiDetectionDataset(cfg.data_root, cfg.DATASET_CONFIG.TRAIN_SET)
    stat_dataset(dataset, 'Training set')
    dataset = KittiDetectionDataset(cfg.data_root, cfg.DATASET_CONFIG.VAL_SET)
    stat_dataset(dataset, 'Validation set')