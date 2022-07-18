from copy import deepcopy
from datetime import datetime
from tqdm import tqdm
from datasets.kitti_detection import KittiDetectionDataset
from datasets.kitti_loader import build_kitti_loader, move_to_cuda, merge_two_batch
import yaml
from easydict import EasyDict
import argparse
import torch
from os import path, makedirs
from models.MTrans import MTrans
from torch.utils.tensorboard import SummaryWriter
from utils.lr_scheduler import WarmupCosineAnnealing
import random
import numpy as np
from utils.stat_scores import HistoCounter, ScoreCounter
import pickle

def freeze_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def save_checkpoint(save_path, epoch, model, optim=None, scheduler=None):
    if not path.exists(path.dirname(save_path)):
        makedirs(path.dirname(save_path))
    print(f">>> Saving checkpoint as: {save_path}")
    model_state_dict = model.state_dict()
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model_state_dict
    }
    if optim is not None:
        ckpt['optimizer_state_dict'] = optim.state_dict()
    if scheduler is not None:
        ckpt['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(ckpt, save_path)

def load_checkpoint(file_path, model, optim=None, scheduler=None):
    ckpt = torch.load(file_path)
    model_ckpt = ckpt['model_state_dict']
    model.load_state_dict(model_ckpt)
    if optim is not None:
        optim.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    epoch = ckpt['epoch']
    return epoch

def get_pbar_text(counter, prefix):
    stats = counter.average(['loss', 'iou3d', 'segment_iou', 'err_dist', 'recall_7', 'acc_dir', 'err_conf'])
    pbar_text = f"{prefix} L:{stats['loss']:.2f}, Seg:{stats['segment_iou']*100:.2f}, XYZ:{stats['err_dist']:.2f}, IoU:{stats['iou3d']*100:.2f}, R:{stats['recall_7']*100:.2f}, Dr:{stats['acc_dir']*100:.2f}, Cf: {stats['err_conf']*100:.2f}"
    return pbar_text

def train_one_epoch(cfg,
                    model, 
                    training_loader, 
                    unlabeled_training_loader,
                    optim, 
                    scheduler, 
                    counter, 
                    histo_counter,
                    epoch, 
                    writer):
    model.train()
    if unlabeled_training_loader is not None:
        process_bar = tqdm(training_loader, desc='E{epoch}')
        unlabeled_iter = iter(unlabeled_training_loader)
    else:
        process_bar = tqdm(training_loader, desc='E{epoch}')
    counter.reset()      
    histo_counter.reset()  

    for data in process_bar:
        optim.zero_grad()
        data = EasyDict(data)
        if unlabeled_training_loader is not None:
            try:
                unlabeled_data = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_training_loader)
                unlabeled_data = next(unlabeled_iter)
            unlabeled_data = EasyDict(unlabeled_data)
            data = merge_two_batch(data, unlabeled_data)
        data = move_to_cuda(data, 'cuda')
        pred_dict = model(data)
        loss_dict, loss, loss_segment, loss_depth, loss_box, iou3d_histo = model.get_loss(pred_dict, data)
        histo_counter.update(iou3d_histo)

        # statistics
        counter.update(loss_dict)

        loss.backward()
        optim.step()
        scheduler.step()
        counter.update({'lr':(optim.param_groups[0]['lr'], 1, 'learning_rate')})

        pbar_text = get_pbar_text(counter, f'T-{epoch}')        
        process_bar.set_description(pbar_text)

    stats = counter.average(None, group_by_description=True)
    for group in stats.keys():
        writer.add_scalars(f'Train/{group}', stats[group], epoch)
    writer.add_histogram('Train/iou_distribution', histo_counter.get_values(), epoch)

def eval(cfg, model, validation_loader, counter, histo_counter, epoch, writer):
    model.eval()
    process_bar = tqdm(validation_loader, desc='Evaluate model')
    counter.reset()
    histo_counter.reset()
    all_nuscenes_boxes = {}
    with torch.no_grad():
        for data in process_bar:
            data = EasyDict(data)
            data = move_to_cuda(data, 'cuda')
            pred_dict = model(data)
            loss_dict, loss, loss_segment, loss_depth, loss_box, iou3d_histo = model.get_loss(pred_dict, data)

            if cfg.gen_label and cfg.dataset=='KITTI':
                label, frames = model.format_kitti_labels(pred_dict, data, with_score=(validation_loader.dataset.cfg.split=='test'))
                if not path.exists(f'{cfg.label_dir}'):
                    makedirs(f'{cfg.label_dir}')
                for i, fr in enumerate(frames):
                    with open(path.join(f'{cfg.label_dir}', f'{fr}.txt'), 'a') as f:
                        l = label[i]
                        # score = float(l.split(' ')[-1])       # [optional]: discard 3D predictions with low confidence
                        # if score<0.05:
                        #     continue
                        f.write(l+'\n')  

            # statistics
            counter.update(loss_dict)
            histo_counter.update(iou3d_histo)

            pbar_text = get_pbar_text(counter, f'Eval')
            process_bar.set_description(pbar_text)

        stats = counter.average(None, group_by_description=True)
        for group in stats.keys():
            writer.add_scalars(f'Eval/{group}', stats[group], epoch)
        writer.add_histogram('Eval/iou_distribution', histo_counter.get_values(), epoch)
        
        # metric for saving best checkpoint
        score = (counter.average(['iou3d'])['iou3d'])

    return score


def main(cfg, cfg_path):
    # Tensorboard, Yaml Config, Score Counter
    output_path = path.join(cfg.TRAIN_CONFIG.output_root, cfg.experiment_name)
    writer = SummaryWriter(log_dir=path.join(output_path, cfg.experiment_name+'_tb'))
    writer.add_text('experiment_name', cfg.experiment_name, 0)
    writer.add_text('start_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 0)
    counter = ScoreCounter()
    histo_counter = HistoCounter()

    train_cfg, dataset_cfg, loader_cfg = cfg.TRAIN_CONFIG, cfg.DATASET_CONFIG, cfg.DATALOADER_CONFIG
    data_root = cfg.data_root
    # build dataset and dataloader
    if cfg.dataset == 'KITTI':
        nusc = None
        dataset = KittiDetectionDataset
        loader_builder = build_kitti_loader
    else:
        raise RuntimeError
    training_set = dataset(data_root, dataset_cfg.TRAIN_SET, nusc=nusc)
    training_loader = loader_builder(training_set, loader_cfg.TRAIN_LOADER)
    train_length = len(training_loader)
    if loader_cfg.TRAIN_LOADER.unsupervise_batch_size > 0:
        # build another dataset that has no 3D label
        temp_cfg = deepcopy(dataset_cfg.TRAIN_SET)
        temp_cfg.use_3d_label = False
        unlabeled_training_set = dataset(data_root, temp_cfg, nusc=nusc)
        # training loader for unlabeled dataset
        temp_cfg = deepcopy(loader_cfg.TRAIN_LOADER)
        temp_cfg.batch_size = temp_cfg.unsupervise_batch_size
        unlabeled_training_loader = loader_builder(unlabeled_training_set, temp_cfg)
    else:
        unlabeled_training_loader = None
        
    validation_set = dataset(data_root, dataset_cfg.VAL_SET, nusc=nusc)
    validation_loader = loader_builder(validation_set, loader_cfg.VAL_LOADER)

    # build model
    model = MTrans(cfg.MODEL_CONFIG)
    model.cuda()
    print("[Model Params]: ", sum([p.numel() for n, p in model.named_parameters()]))
    writer.add_text('model_params', str(sum([p.numel() for n, p in model.named_parameters()])))

    # build optimizer and lr_scheduler
    optim = getattr(torch.optim, train_cfg.optimizer)(lr=train_cfg.lr, params=model.parameters())
    scheduler = WarmupCosineAnnealing(optim, train_cfg.lr, train_cfg.warmup_rate, train_cfg.epochs*train_length, eta_min=0)
    scheduler.step()

    # load checkpoint
    start_epoch=0
    best_score = -9999
    if cfg.init_checkpoint is not None:
        print(f"Loading checkpoint at: {cfg.init_checkpoint}")
        start_epoch = load_checkpoint(f'{cfg.init_checkpoint}', model, optim, scheduler) + 1
    elif path.exists(f'{cfg.TRAIN_CONFIG.output_root}/{cfg.experiment_name}/ckpt/best_model.pt'):
        print("Loading best checkpoints...")
        start_epoch = load_checkpoint(f'{cfg.TRAIN_CONFIG.output_root}/{cfg.experiment_name}/ckpt/best_model.pt', model, optim, scheduler) + 1

    if start_epoch > 0:
        best_score = eval(cfg, model, validation_loader, counter, histo_counter, start_epoch-1, writer)   
    
    for epoch in range(start_epoch, train_cfg.epochs):
        ### TRAINING ###
        train_one_epoch(cfg, model, training_loader, unlabeled_training_loader, optim, scheduler, counter, histo_counter, epoch, writer)

        ### EVALUATION ###
        if ((epoch+1) % train_cfg.epoches_per_eval) == 0 and epoch >= train_cfg.eval_begin:
            score = eval(cfg, model, validation_loader, counter, histo_counter, epoch, writer)            
            if score > best_score:
                best_score = score
                save_checkpoint(f'{cfg.TRAIN_CONFIG.output_root}/{cfg.experiment_name}/ckpt/best_model.pt', epoch, model, optim, scheduler)

    if start_epoch < train_cfg.epochs:
    # save last checkpoint
        save_checkpoint(f'{cfg.TRAIN_CONFIG.output_root}/{cfg.experiment_name}/ckpt/epoch_{epoch}.pt', epoch, model, optim, scheduler)

    writer.flush()
    writer.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='training arguments')
    parser.add_argument('--cfg_file', type=str, help='the path to configuration file')
    args = parser.parse_args()
    cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))

    print("===== START TRAINING =====")
    print(cfg.experiment_name)
    print("==========================")

    freeze_random_seed(cfg.random_seed)
    main(cfg, args.cfg_file)