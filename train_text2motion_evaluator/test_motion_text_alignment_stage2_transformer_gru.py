## How to run: python -m torch.distributed.launch --nproc_per_node=2 train_vq_vae_tokenizer_motion.py
##  CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=29999 test_motion_text_alignment_stage2_transformer_gru.py

from __future__ import print_function
import os
import time
import argparse
import datetime

import numpy as np
from os.path import join as pjoin
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from dataset_text2motion_new_joints_vec import prepare_text2motion_data

from motion_text_alignment_model_transformer_gru import Motion_Text_Alignment_Model

import options as opt
from info_nce import InfoNCE

## transformer autoencoder
## transformer encoder
encoder_latent_dim=int(768)
encoder_ff_size=int(1024)
encoder_num_layers=int(6)
encoder_num_heads=int(8)
encoder_dropout=float(0.1)
encoder_activation='gelu'

## transformer decoder
decoder_latent_dim=int(768)
decoder_ff_size=int(1024)
decoder_num_layers=int(6)
decoder_num_heads=int(8)
decoder_dropout=float(0.1)
decoder_activation='gelu'

def init_distributed_mode(args):
    '''initilize DDP 
    '''
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}, local rank:{args.gpu}, world size:{args.world_size}", flush=True)
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    return args.gpu

# padding to max length in one batch
def collate_tensors(batch):
    if isinstance(batch[0], np.ndarray):
        batch = [torch.tensor(b).float() for b in batch]

    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def collate_fn(batch):
    notnone_batches = [b for b in batch if b is not None]
    adapted_batch = {'motion_feats': collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
                     'texts': [b[0] for b in notnone_batches],
                     'motion_masks': collate_tensors([b[2] for b in notnone_batches])
    }
    return adapted_batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()

    ## use cuda
    use_cuda = torch.cuda.is_available()
    gpus = init_distributed_mode(args)
    device = torch.device("cuda" if use_cuda else "cpu")

    ## loading motion data
    # dataset_train = prepare_text2motion_data('t2m', 'train')
    # dataset_val = prepare_text2motion_data('t2m', 'val')
    dataset_test = prepare_text2motion_data('t2m', 'test')

    ## define distributed sampler
    # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train) 
    # val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val)
    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)

    # train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, 1, drop_last=True)

    ## define dataloader
    # train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_sampler=train_batch_sampler, pin_memory=True, num_workers=opt.num_workers, collate_fn=collate_fn)
    # val_dataloader = torch.utils.data.DataLoader(dataset_val, sampler=val_sampler, pin_memory=True, num_workers=opt.num_workers, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(dataset_test, sampler=test_sampler, pin_memory=True, num_workers=opt.num_workers, collate_fn=collate_fn)

    ## define motion transformer ae model
    motion_text_alignment_model = Motion_Text_Alignment_Model(
        ## nfeats
        263, 

        ## transformer encoder parameters
        259,
        512,
        512, 

        ## transformer decoder parameters
        512,
        1024,
        512, 

        device,
    )
    model = motion_text_alignment_model

    # print(model)
    ## loading trained motion-text alignment model
    if os.path.exists(os.path.join(opt.motion_text_alignment_pretrained_dir, 'pretrained_motion_text_alignment_512_movement_bigruco_v6.pt')):
        pretrained_motion_text_alignment_weight_file = os.path.join(opt.motion_text_alignment_pretrained_dir, "pretrained_motion_text_alignment_512_movement_bigruco_v6.pt")
        state_dict = torch.load(pretrained_motion_text_alignment_weight_file, map_location='cpu')
        
        if 'module' in list(state_dict.keys())[0]:
            from collections import OrderedDict
            motion_text_alignment_dict = OrderedDict()
            # import pdb; pdb.set_trace()
            for k,v in state_dict.items():
                name = k.replace('module.', '')
                motion_text_alignment_dict[name] = v
            model.load_state_dict(motion_text_alignment_dict, strict=True)
        else:
            model.load_state_dict(state_dict, strict=True)
        print('Loading pretrained motion-text alignment model successfully.')

    else:
        raise ValueError("pretrained_motion_text_alignment_512_movement_bigruco_v6.pt must be exist.")

    model = model.to(device)

    # ## ddp model
    # if opt.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpus], find_unused_parameters=True)
    #     model_without_ddp = model.module

    model.eval()
    # for batch_idx, batch in enumerate(train_dataloader):
    #     m_feats = batch['motion_feats'].to(device)
    #     texts = batch['texts']
    #     masks = batch['motion_masks'].to(device)
    #     m_avg, t_avg, m_decoder_outputs, t_decoder_outputs = model(m_feats, texts, masks)
    #     similarity = torch.cosine_similarity(m_avg[0], t_avg[0], dim=0)
    #     print(f'train sample {batch_idx} similarity: ', similarity)
    #     # import pdb; pdb.set_trace()

    # for batch_idx, batch in enumerate(tqdm(val_dataloader)):
    #     m_feats = batch['motion_feats'].to(device)
    #     texts = batch['texts']
    #     masks = batch['motion_masks'].to(device)
    #     m_avg, t_avg, m_decoder_outputs, t_decoder_outputs = model(m_feats, texts, masks)
    #     # import pdb; pdb.set_trace()
    #     similarity = torch.cosine_similarity(m_avg[0], t_avg[0], dim=0)
    #     print(f'val sample {batch_idx} similarity: ', similarity)

    for batch_idx, batch in enumerate(tqdm(test_dataloader)):
        m_feats = batch['motion_feats'].to(device)
        texts = batch['texts']
        masks = batch['motion_masks'].to(device)
        # m_avg, t_avg, m_decoder_outputs, t_decoder_outputs = model(m_feats, texts, masks)
        m_avg, t_avg = model(m_feats, texts, masks)
        similarity = torch.cosine_similarity(m_avg[0], t_avg[0], dim=0)
        print(f'test sample {batch_idx} similarity: ', similarity)

if __name__ == '__main__':
    start = time.time()
    main()
    print(f'Total cost time:{time.time() - start} ms')