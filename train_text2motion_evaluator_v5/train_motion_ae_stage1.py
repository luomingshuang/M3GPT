## How to run: python -m torch.distributed.launch --nproc_per_node=2 train_vq_vae_tokenizer_motion.py
##  CUDA_VISIBLE_DEVICES='0' torchrun --nproc_per_node=1 --master_port=29999 train_text_motion_clip.py

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

from dataset_text2motion import prepare_text2motion_data

from motion_transformer_ae_model import Motion_Text_Alignment_Model

import options as opt


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
    dataset_train = prepare_text2motion_data('t2m', 'train')
    dataset_val = prepare_text2motion_data('t2m', 'val')
    dataset_test = prepare_text2motion_data('t2m', 'test')

    ## define distributed sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train) 
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val)
    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)

    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, opt.batch_size, drop_last=True)

    ## define dataloader
    train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_sampler=train_batch_sampler, pin_memory=True, num_workers=opt.num_workers, collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(dataset_val, sampler=val_sampler, pin_memory=True, num_workers=opt.num_workers, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(dataset_test, sampler=test_sampler, pin_memory=True, num_workers=opt.num_workers, collate_fn=collate_fn)

    ## define motion transformer ae model
    motion_tae = Motion_Text_Alignment_Model(
        ## nfeats
        263, 

        ## transformer encoder parameters
        opt.encoder_latent_dim,
        opt.encoder_ff_size,
        opt.encoder_num_layers,
        opt.encoder_num_heads,
        opt.encoder_dropout,
        opt.encoder_activation,

        ## transformer decoder parameters
        opt.decoder_latent_dim,
        opt.decoder_ff_size,
        opt.decoder_num_layers,
        opt.decoder_num_heads,
        opt.decoder_dropout,
        opt.decoder_activation,

        device
    )
    model = motion_tae

    ## loading trained motion-ae model
    if opt.load_pretrained_model:
        if not os.path.exists(opt.motion_text_alignment_pretrained_dir): os.mkdir(opt.motion_text_alignment_pretrained_dir)
        if os.path.exists(os.path.join(opt.motion_text_alignment_pretrained_dir, opt.weight_file_name)):
            pretrained_motion_tae_weight_file = os.path.join(opt.motion_text_alignment_pretrained_dir, opt.weight_file_name)
            state_dict = torch.load(pretrained_motion_tae_weight_file, map_location='cpu')
            if 'module' in list(state_dict.keys())[0]:
                from collections import OrderedDict
                vae_dict = OrderedDict()
                for k,v in state_dict.items():
                    name = k.replace('module.', '')
                    vae_dict[name] = v
                model.load_state_dict(vae_dict, strict=False)
            else:
                from collections import OrderedDict
                vae_dict = OrderedDict()
                for k,v in state_dict.items():
                    vae_dict[name] = v
                model.load_state_dict(state_dict, strict=False)
            print('Loading pretrained motion tae model successfully.')

    model = model.to(device)

    ## ddp model
    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpus], find_unused_parameters=True)
        model_without_ddp = model.module

    ## define optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=opt.lr, betas=opt.betas, weight_decay=opt.weight_decay)

    ## lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=opt.T_max, eta_min=opt.eta_min)

    ## define loss
    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()

    ## define tensorboard log
    def check_print_rank(params):
        return params.world_size == 1 or torch.distributed.get_rank() == 0
    if check_print_rank(args):
        start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tb_writer = SummaryWriter(log_dir="tensorboard_log_motion_tae/log-train-{}".format(start_time))
    else:
        tb_writer = None

    iteration = 0
    for epoch in range(1, opt.max_epoch):
        model.train()
        if opt.distributed:
            train_sampler.set_epoch(epoch)

        tot_train_loss = 0
        tot_train_batches = 0
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            m_feats = batch['motion_feats'].to(device)
            texts = batch['texts']
            masks = batch['motion_masks'].to(device)
            m_feats, m_decoder_outputs = model(m_feats, masks)

            optimizer.zero_grad()
            loss = criterion(m_feats, m_decoder_outputs)
            loss.backward()
            optimizer.step()

            tot_train_loss += loss.detach().cpu().numpy()
            tot_train_batches += 1

            ## add train tensorboard log
            if tb_writer is not None:
                tb_writer.add_scalar('train/loss_iteration', loss.detach().cpu().numpy(), iteration)
            iteration += 1

        lr_scheduler.step()

        ## save model
        if opt.distributed:
            if torch.distributed.get_rank() == 0:
                # only save model on GPU0 process.
                torch.save(model.state_dict(), "{}/{}".format(opt.motion_text_alignment_pretrained_dir, opt.weight_file_name))
                # if epoch % 100 == 0:
                #     torch.save(model.state_dict(), "{}/pretrained_motion_tae_{}.pt".format(opt.motion_text_alignment_pretrained_dir, epoch))
        else:
            torch.save(model.state_dict(), "{}/{}".format(opt.motion_text_alignment_pretrained_dir, opt.weight_file_name))
            # if epoch % 100 == 0:
            #     torch.save(model.state_dict(), "{}/pretrained_motion_tae_{}.pt".format(opt.motion_text_alignment_pretrained_dir, epoch))
        
        ## add train tensorboard log
        train_average_loss = tot_train_loss / tot_train_batches
        if tb_writer is not None:
            tb_writer.add_scalar('train/loss_epoch', train_average_loss, epoch)
        print('{} epoch, l2 train total average loss is {}'.format(epoch, train_average_loss))

        if epoch % 50 == 0:
            ## compute validation loss
            model.eval()
            tot_val_loss = 0
            tot_val_batches = 0
            
            for batch_idx, batch in enumerate(tqdm(val_dataloader)):
                m_feats = batch['motion_feats'].to(device)
                texts = batch['texts']
                masks = batch['motion_masks'].to(device)
                m_feats, m_decoder_outputs = model(m_feats, masks)
                
                loss = criterion(m_feats, m_decoder_outputs)

                tot_val_loss += loss.detach().cpu().numpy()
                tot_val_batches += 1
            val_average_loss = tot_val_loss / tot_val_batches

            ## add val tensorboard log
            if tb_writer is not None:
                tb_writer.add_scalar('val/loss_epoch', val_average_loss, epoch)
            print('{} epoch, l2 val total average loss is {}'.format(epoch, val_average_loss))
    
            ## compute test loss
            tot_test_loss = 0
            tot_test_batches = 0

            for batch_idx, batch in enumerate(tqdm(test_dataloader)):
                m_feats = batch['motion_feats'].to(device)
                texts = batch['texts']
                masks = batch['motion_masks'].to(device)
                m_feats, m_decoder_outputs = model(m_feats, masks)

                loss = criterion(m_feats, m_decoder_outputs)

                tot_test_loss += loss.detach().cpu().numpy()
                tot_test_batches += 1
            test_average_loss = tot_test_loss / tot_test_batches

            ## add test tensorboard log
            if tb_writer is not None:
                tb_writer.add_scalar('test/loss_epoch', test_average_loss, epoch)
            print('{} epoch, l2 test total average loss is {}'.format(epoch, test_average_loss))

if __name__ == '__main__':
    start = time.time()
    main()
    print(f'Total cost time:{time.time() - start} ms')