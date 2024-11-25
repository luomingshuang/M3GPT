# running command:
# python encode_wav_to_tokens.py --model=5b --load_path=./models --result_path=./audio_result_motiongpt_d2a_finedance --model_level=high

import os
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import time
import argparse
from pathlib import Path
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

import torch
from librosa.core import load
from librosa.util import normalize

def load_wav_to_torch(full_path, sampling_rate=22050, augment=False):
    """
    Loads wavdata into torch array
    """
    data, sampling_rate = load(full_path, sr=sampling_rate)
    data = 0.95 * normalize(data)

    if augment:
        amplitude = np.random.uniform(low=0.3, high=1.0)
        data = data * amplitude

    return torch.from_numpy(data).float(), sampling_rate

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
        vqvae.load_state_dict(t.load("./models/vqvae_high.pt"))
        vqvae.eval()
    if model_level == "low":
        vqvae.load_state_dict(t.load("./models/vqvae_low.pt"))
        vqvae.eval()

    print("*******Finish model loading******")

    wav_file = '/home/luomingshuang/data/dataset_3dmotion/finedance/music_wav/002.wav'
    audio_data, sr = load_wav_to_torch(wav_file, sampling_rate=22050)
    audio_data = audio_data
    audio_data = audio_data.unsqueeze(0).float().cuda()
    audio_data = audio_data.unsqueeze(0)
    gt_xs, zs_code = vqvae._encode(audio_data.transpose(1,2))
    zs_middle = []
    zs_middle.append(zs_code[code_level])

    quantised_xs, out = vqvae._decode(zs_middle, start_level=level_s, end_level=level_e)

    # original samples
    name = wav_file.split('/')[-1][:-4]
    audio = audio_data.squeeze().cpu()
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    sample_original = 'original_' + name + '.wav'
    sample_vqvae = 'vqvae_'+ name + '.wav'
    sample_vqvae_tokens = 'music_vqvae_tokens_' + name + '.npy' 
    sample_original = os.path.join(result_path,sample_original)
    sample_vqvae = os.path.join(result_path,sample_vqvae) 
    sf.write(sample_original, audio.detach().cpu().numpy(), 22050)
    sf.write(sample_vqvae, out.squeeze().detach().cpu().numpy(), 22050)
    np.save(os.path.join(result_path, sample_vqvae_tokens), zs_middle[0].detach().cpu().numpy())

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
