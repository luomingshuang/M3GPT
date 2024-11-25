## How to run: python -m torch.distributed.launch --nproc_per_node=2 train_vq_vae_tokenizer_motion.py
##  CUDA_VISIBLE_DEVICES='0' torchrun --nproc_per_node=1 --master_port=29999 extract_motion_cb.py

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

from datasets.dataset_motion import prepare_motion_data

from options import train_vq_vae_motion_options as opt

from archs.mgpt_vq import VQVae

from losses.mgpt import GPTLosses


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
    adapted_batch = {
        'name': [b[0] for b in notnone_batches],
        'motion': collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
        'length': [b[2] for b in notnone_batches],
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
    motion_vq_dataset_train, motion_vq_dataset_val, motion_vq_dataset_test = prepare_motion_data()

    ## define distributed sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(motion_vq_dataset_train) 
    val_sampler = torch.utils.data.distributed.DistributedSampler(motion_vq_dataset_val)
    test_sampler = torch.utils.data.distributed.DistributedSampler(motion_vq_dataset_test)

    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, 1, drop_last=True)

    ## define dataloader
    train_dataloader = torch.utils.data.DataLoader(motion_vq_dataset_train, batch_sampler=train_batch_sampler, pin_memory=True, num_workers=opt.num_workers, collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(motion_vq_dataset_val, sampler=val_sampler, pin_memory=True, num_workers=opt.num_workers, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(motion_vq_dataset_test, sampler=test_sampler, pin_memory=True, num_workers=opt.num_workers, collate_fn=collate_fn)
    
    ## define vq_vae motion model
    motion_vae_model = VQVae(
        nfeats=opt.nfeats,
        quantizer=opt.quantizer,
        code_num=opt.code_num,
        code_dim=opt.code_dim,
        output_emb_width=opt.output_emb_width,
        down_t=opt.down_t,
        stride_t=opt.stride_t,
        width=opt.width,
        depth=opt.depth,
        dilation_growth_rate=opt.dilation_growth_rate,
        norm=opt.norm,
        activation=opt.activation,
    )

    model = motion_vae_model

    ## loading trained motion-vae model
    state_dict = torch.load("/data/luomingshuang/checkpoints/unified_io_motion_tasks/vq_vae_pretrained_motion/unified_vq_vae_motion_20240306.pt", map_location='cpu')
    if 'module' in list(state_dict.keys())[0]:
        from collections import OrderedDict
        vae_dict = OrderedDict()
        for k,v in state_dict.items():
            name = k.replace('module.', '')
            vae_dict[name] = v
        model.load_state_dict(vae_dict, strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)
    print('Loading pretrained motion vqvae model successfully.')
    model = model.to(device)

    ## ddp model
    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(motion_vae_model, device_ids=[gpus])
        model_without_ddp = model.module

    ## define tensorboard log
    # def check_print_rank(params):
    #     return params.world_size == 1 or torch.distributed.get_rank() == 0
    # if check_print_rank(args):
    #     start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #     #tb_writer = SummaryWriter(log_dir="tensorboard_log/log-train-{}".format(start_time))
    # else:
    #     tb_writer = None

    with torch.no_grad():
        model.eval()
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            name = batch['name'][0]
            # name = name.replace('/data/luomingshuang/data/dataset_3dmotion/motionx_finedance_aistpp_30fps_22joints', opt.vqvae_motion_save_path)
            name = name.replace('motionx_finedance_aistpp_30fps_22joints', 'motionx_finedance_aistpp_30fps_22joints_vqvae_rec')
            name_dir = '/'.join(name.split('/')[:-1])
            if not os.path.exists(name_dir):
                os.makedirs(name_dir)

            feats_ref = batch['motion'].to(device)

            ## motion encode feats to codes idx
            code_idxs, _ = model.module.encode(feats_ref)
            feats_rst = model.module.decode(code_idxs)
            # import pdb; pdb.set_trace()
            np.save(name+'.npy', feats_rst[0].detach().cpu().numpy())

        for batch_idx, batch in enumerate(tqdm(val_dataloader)):
            name = batch['name'][0]
            # name = name.replace('/data/luomingshuang/data/dataset_3dmotion/motionx_finedance_aistpp_30fps_22joints', opt.vqvae_motion_save_path)
            name = name.replace('motionx_finedance_aistpp_30fps_22joints', 'motionx_finedance_aistpp_30fps_22joints_vqvae_rec')
            name_dir = '/'.join(name.split('/')[:-1])
            if not os.path.exists(name_dir):
                os.makedirs(name_dir)

            feats_ref = batch['motion'].to(device)

            ## motion encode feats to codes idx
            code_idxs, _ = model.module.encode(feats_ref)
            feats_rst = model.module.decode(code_idxs)
            np.save(name+'.npy', feats_rst[0].detach().cpu().numpy())

        for batch_idx, batch in enumerate(tqdm(test_dataloader)):
            name = batch['name'][0]
            # name = name.replace('/data/luomingshuang/data/dataset_3dmotion/motionx_finedance_aistpp_30fps_22joints', opt.vqvae_motion_save_path)
            name = name.replace('motionx_finedance_aistpp_30fps_22joints', 'motionx_finedance_aistpp_30fps_22joints_vqvae_rec')
            name_dir = '/'.join(name.split('/')[:-1])
            if not os.path.exists(name_dir):
                os.makedirs(name_dir)

            feats_ref = batch['motion'].to(device)

            ## motion encode feats to codes idx
            code_idxs, _ = model.module.encode(feats_ref)
            feats_rst = model.module.decode(code_idxs)
            np.save(name+'.npy', feats_rst[0].detach().cpu().numpy())

if __name__ == '__main__':
    start = time.time()
    main()
    print(f'Total cost time:{time.time() - start} ms')