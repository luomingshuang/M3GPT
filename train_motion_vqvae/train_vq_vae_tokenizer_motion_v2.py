## How to run: python -m torch.distributed.launch --nproc_per_node=2 train_vq_vae_tokenizer_motion.py
##  CUDA_VISIBLE_DEVICES='0,1' torchrun --nproc_per_node=2 --master_port=29999 train_vq_vae_tokenizer_motion.py

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

from datasets.dataset_motion_vq_v2 import prepare_motion_vqv2_data

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
    adapted_batch = {'motion': collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
                     'length': [b[1] for b in notnone_batches],
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
    motion_vq_dataset_train, motion_vq_dataset_val, motion_vq_dataset_test = prepare_motion_vqv2_data()

    ## define distributed sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(motion_vq_dataset_train) 
    val_sampler = torch.utils.data.distributed.DistributedSampler(motion_vq_dataset_val)
    test_sampler = torch.utils.data.distributed.DistributedSampler(motion_vq_dataset_test)

    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, opt.batch_size*10, drop_last=True)
    val_batch_sampler = torch.utils.data.BatchSampler(val_sampler, opt.batch_size*10, drop_last=True)
    test_batch_sampler = torch.utils.data.BatchSampler(test_sampler, opt.batch_size*10, drop_last=True)


    ## define dataloader
    train_dataloader = torch.utils.data.DataLoader(motion_vq_dataset_train, batch_sampler=train_batch_sampler, pin_memory=True, num_workers=opt.num_workers, collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(motion_vq_dataset_val, batch_sampler=val_batch_sampler, pin_memory=True, num_workers=opt.num_workers, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(motion_vq_dataset_test, batch_sampler=test_batch_sampler, pin_memory=True, num_workers=opt.num_workers, collate_fn=collate_fn)
    
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
    ).to(device)

    model = motion_vae_model
    
    ## ddp model
    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(motion_vae_model, device_ids=[gpus])
        model_without_ddp = model.module

    ## define optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=opt.lr, betas=opt.betas, weight_decay=opt.weight_decay)

    ## lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=opt.T_max, eta_min=opt.eta_min)

    ## define loss
    losses = torch.nn.ModuleDict({split: GPTLosses(opt, 'vae', 22) for split in ["losses_train", "losses_test", "losses_val"]})

    ## define tensorboard log
    def check_print_rank(params):
        return params.world_size == 1 or torch.distributed.get_rank() == 0
    if check_print_rank(args):
        start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tb_writer = SummaryWriter(log_dir="tensorboard_log/log-train-{}-vqvae-v2".format(start_time))
    else:
        tb_writer = None

    for epoch in range(1, opt.max_epoch):
        model.train()
        if opt.distributed:
            train_sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            feats_ref = batch['motion'].to(device)
            joints_ref = motion_vq_dataset_train.feats2joints(feats_ref)
            ## motion encode and decode
            feats_rst, loss_commit, perplexity = model(feats_ref)
            joints_rst = motion_vq_dataset_train.feats2joints(feats_rst)
            rs_set = {
                "m_ref": feats_ref,
                "joints_ref": joints_ref,
                "m_rst": feats_rst,
                "joints_rst": joints_rst,
                "loss_commit": loss_commit,
                "perplexity": perplexity,       
            }

            optimizer.zero_grad()
            loss = losses["losses_train"].update(rs_set)
            loss.backward()
            optimizer.step()
        
        lr_scheduler.step()

        ## save model
        if opt.distributed:
            if torch.distributed.get_rank() == 0:
                # only save model on GPU0 process.
                torch.save(model.state_dict(), "{}/unified_vq_vae_motion.pt".format(opt.vq_vae_motion_pretrained_dir_v2))
                if epoch % 100 == 0:
                    torch.save(model.state_dict(), "{}/unified_vq_vae_motion_{}.pt".format(opt.vq_vae_motion_pretrained_dir_v2, epoch))
        else:
            torch.save(model.state_dict(), "{}/unified_vq_vae_motion.pt".format(opt.vq_vae_motion_pretrained_dir_v2))
            if epoch % 100 == 0:
                torch.save(model.state_dict(), "{}/unified_vq_vae_motion_{}.pt".format(opt.vq_vae_motion_pretrained_dir_v2, epoch))
        ## add train tensorboard log
        if tb_writer is not None:
            tb_writer.add_scalar('train/loss_epoch', loss.detach().cpu().numpy(), epoch)

        if epoch % 50 == 0:
            ## compute validation loss
            model.eval()
            tot_val_loss = 0
            tot_val_batches = 0
            
            for batch_idx, batch in enumerate(tqdm(val_dataloader)):
                feats_ref = batch['motion'].to(device)
                joints_ref = motion_vq_dataset_val.feats2joints(feats_ref)
                ## motion encode and decode
                feats_rst, loss_commit, perplexity = model(feats_ref)
                joints_rst = motion_vq_dataset_val.feats2joints(feats_rst)
                rs_set = {
                    "m_ref": feats_ref,
                    "joints_ref": joints_ref,
                    "m_rst": feats_rst,
                    "joints_rst": joints_rst,
                    "loss_commit": loss_commit,
                    "perplexity": perplexity,       
                }

                loss = losses["losses_val"].update(rs_set)
                tot_val_loss += loss.detach().cpu().numpy()
                tot_val_batches += 1
            val_average_loss = tot_val_loss / tot_val_batches

            ## add val tensorboard log
            if tb_writer is not None:
                tb_writer.add_scalar('val/loss_epoch', val_average_loss, epoch)
    
            ## compute test loss
            tot_test_loss = 0
            tot_test_batches = 0

            for batch_idx, batch in enumerate(tqdm(test_dataloader)):
                feats_ref = batch['motion'].to(device)
                joints_ref = motion_vq_dataset_test.feats2joints(feats_ref)
                ## motion encode and decode
                feats_rst, loss_commit, perplexity = model(feats_ref)
                joints_rst = motion_vq_dataset_test.feats2joints(feats_rst)
                rs_set = {
                    "m_ref": feats_ref,
                    "joints_ref": joints_ref,
                    "m_rst": feats_rst,
                    "joints_rst": joints_rst,
                    "loss_commit": loss_commit,
                    "perplexity": perplexity,       
                }

                loss = losses["losses_test"].update(rs_set)
                tot_test_loss += loss.detach().cpu().numpy()
                tot_test_batches += 1
            test_average_loss = tot_test_loss / tot_test_batches

            ## add test tensorboard log
            if tb_writer is not None:
                tb_writer.add_scalar('test/loss_epoch', test_average_loss, epoch)


if __name__ == '__main__':
    start = time.time()
    main()
    print(f'Total cost time:{time.time() - start} ms')
