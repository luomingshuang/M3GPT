import os
import numpy as np
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import time
import argparse
from pathlib import Path
from glob import glob
import jukebox.utils.dist_adapter as dist

from jukebox.hparams import Hyperparams
from jukebox.utils.torch_utils import empty_cache
from jukebox.utils.audio_utils import save_wav, load_audio
from jukebox.make_models import make_vae_model
from jukebox.utils.sample_utils import split_batch, get_starts
from jukebox.utils.dist_utils import print_once
import fire
import librosa
import soundfile as sf
import noisereduce as nr

from d2m.dataset import VAMDataset
from d2m.d2m_modules import vqEncoder_high,vqEncoder_low, Discriminator, motion_encoder, Audio2Mel
from d2m.utils import save_sample


def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--save_path", required=True)
    parser.add_argument("--load_path", default='./logs')
    parser.add_argument("--model", default='5b')
    parser.add_argument("--result_path", required=True)
    parser.add_argument("--model_level", required=True)

    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=3)

    parser.add_argument("--ndf", type=int, default=32)
    parser.add_argument("--num_D", type=int, default=3)
    parser.add_argument("--n_layers_D", type=int, default=4)
    parser.add_argument("--downsamp_factor", type=int, default=4)
    parser.add_argument("--lambda_feat", type=float, default=10)
    parser.add_argument("--cond_disc", action="store_true")

    parser.add_argument("--data_path", default=None, type=Path)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--n_test_samples", type=int, default=8)
    args = parser.parse_args()
    return args

# Generate and save samples, alignment, and webpage for visualization.
def generate(model, device, hps):

    args = parse_args()
    load_path = args.load_path
    result_path = args.result_path
    model_level = args.model_level

    if model_level == "high":
        code_level = 2
        seq_len = 44032
        level_s = 2
        level_e = 3
    if model_level == "low":
        code_level = 1
        seq_len = 44096
        level_s = 1
        level_e = 2

    vqvae= make_vae_model(model, device, hps).cuda()
    if model_level == "high":
        encoder = vqEncoder_high().cuda()
        vqvae.load_state_dict(t.load("./models/vqvae_high.pt"))
        vqvae.eval()
        encoder.eval()
    if model_level == "low":
        encoder = vqEncoder_low().cuda()
        vqvae.load_state_dict(t.load("./models/vqvae_low.pt"))
        vqvae.eval()
        encoder.eval()

    mencoder = motion_encoder().cuda()
    mencoder.eval()
    fft = Audio2Mel(n_mel_channels=128).cuda()
    load_path_mencoder = os.path.join(load_path,'mencoder.pt')
    load_path_encoder = os.path.join(load_path,'netG.pt')
    mencoder.load_state_dict(t.load(load_path_mencoder))
    encoder.load_state_dict(t.load(load_path_encoder))
    print("*******Finish model loading******")

    #### generate samples ####
    t.backends.cudnn.benchmark = True
    d2a_generate_code_npy_dir = '/home/luomingshuang/codes/multi-modal-motion-generation/unihcp-for-unified-motion-tasks/test_lm_pretrain/visualizations_new7_test_d2a_finedance'

    vqvae_ref = glob(os.path.join(d2a_generate_code_npy_dir, 'ref', '*.npy'))
    vqvae_rst = glob(os.path.join(d2a_generate_code_npy_dir, 'rst', '*.npy'))

    for sample in vqvae_ref:
        name = sample.split('/')[-1][:-4]
        a_ref = []
        for token in np.load(sample):
            if token != 2048 and token != 2049:
                a_ref.append(token)

        a_rst = []
        for token in np.load(sample.replace('ref', 'rst')):
            if token != 2048 and token != 2049:
                a_rst.append(token)

        a_ref = torch.LongTensor(a_ref).cuda().unsqueeze(0)
        a_rst = torch.LongTensor(a_rst).cuda().unsqueeze(0)
        print(a_ref, a_ref.shape)
        print(a_rst, a_rst.shape)
        quantised_xs_ref, out_ref = vqvae._decode([a_ref], start_level=level_s, end_level=level_e)
        quantised_xs_rst, out_rst = vqvae._decode([a_rst], start_level=level_s, end_level=level_e)
        
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        sample_ref = 'ref_' + name + '.wav'
        sample_rst = 'rst_'+ name + '.wav'
        sample_ref = os.path.join(result_path, sample_ref)
        sample_rst = os.path.join(result_path, sample_rst)

        sf.write(sample_ref, out_ref.squeeze().detach().cpu().numpy(), 22050)
        sf.write(sample_rst, out_rst.squeeze().detach().cpu().numpy(), 22050)
        # gt_code_error = F.l1_loss(gt_xs[code_level], quantised_xs[0])
        # gt_audio_error = F.l1_loss(audio[0:seq_len], out.squeeze().cpu())

        ## reconstructed samples
        # mx = mencoder(m_t)
        # fuse_x = t.cat((mx, v_t),2)
        # pred_xs = encoder(fuse_x, genre, 1)
        # xs_code = []
        # for w in range(3):
        #     xs_code.append(pred_xs)
        # zs_pred = vqvae.bottleneck.encode(xs_code)
        # zs_pred_code = []
        # zs_pred_code.append(zs_pred[code_level])
        # xs_quantised_pred, pred_audio = vqvae._decode(zs_pred_code,start_level=level_s,end_level=level_e)

        # pred_code_error = F.l1_loss(pred_xs,quantised_xs[0] )
        # gen_audio_error = F.l1_loss(audio[0:seq_len], pred_audio.squeeze().cpu())
        # pred_audio = pred_audio.squeeze().detach().cpu().numpy()
        # pred_audio = nr.reduce_noise(y=pred_audio, sr=22050)
        # sample_generated = 'generated_'+ str(i+1) + '.wav'
        # sample_generated = os.path.join(result_path,sample_generated)
        # sf.write(sample_generated, pred_audio, 22050)
        # print("Generating testing sample:", i+1)
    print("*******Finish generating samples*******")

def run(model, mode='ancestral', codes_file=None, audio_file=None, prompt_length_in_seconds=None, port=29500, **kwargs):
    from jukebox.utils.dist_utils import setup_dist_from_mpi
    rank, local_rank, device = setup_dist_from_mpi(port=port)
    hps = Hyperparams(**kwargs)
    # sample_hps = Hyperparams(dict(mode=mode, codes_file=codes_file, audio_file=audio_file, prompt_length_in_seconds=prompt_length_in_seconds))

    with t.no_grad():
        generate(model, device, hps)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    fire.Fire(run)
