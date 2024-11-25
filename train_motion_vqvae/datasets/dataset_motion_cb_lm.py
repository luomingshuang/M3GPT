import os
import rich
import random
import pickle
import codecs as cs
import numpy as np
from tqdm import tqdm

import torch
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin

import sys
sys.path.append('/home/luomingshuang/codes/multi-modal-motion-generation/unified-io-for-any-motion-tasks-lms/datasets')
from datasets import dataset_options as dataset_opt

from scripts.motion_process import process_file, recover_from_ric

## ignore samples, nan
ignore_list = [
    "/data/luomingshuang/data/dataset_3dmotion/motionx_finedance_aistpp_30fps_22joints/motionx_new_joint_vecs/game_motion/subset_0025/Move_Walk",
    "/data/luomingshuang/data/dataset_3dmotion/motionx_finedance_aistpp_30fps_22joints/motionx_new_joint_vecs/game_motion/subset_0006/Emotion_Dogeza_clip_6",
    "/data/luomingshuang/data/dataset_3dmotion/motionx_finedance_aistpp_30fps_22joints/motionx_new_joint_vecs/game_motion/subset_0001/Damage_Walking_clip_12",
    "/data/luomingshuang/data/dataset_3dmotion/motionx_finedance_aistpp_30fps_22joints/motionx_new_joint_vecs/game_motion/subset_0026/Pause_Y",
    "/data/luomingshuang/data/dataset_3dmotion/motionx_finedance_aistpp_30fps_22joints/motionx_new_joint_vecs/game_motion/subset_0026/Pause_Y_clip_4",
    "/data/luomingshuang/data/dataset_3dmotion/motionx_finedance_aistpp_30fps_22joints/motionx_new_joint_vecs/game_motion/subset_0018/Life_Drink",
    "/data/luomingshuang/data/dataset_3dmotion/motionx_finedance_aistpp_30fps_22joints/motionx_new_joint_vecs/dance/subset_0002/King_Kong_Pillow_Tale_Clip1_clip_2",
    "/data/luomingshuang/data/dataset_3dmotion/motionx_finedance_aistpp_30fps_22joints/motionx_new_joint_vecs/humanml/000990",
    "/data/luomingshuang/data/dataset_3dmotion/motionx_finedance_aistpp_30fps_22joints/motionx_new_joint_vecs/humanml/M000990",
    "/data/luomingshuang/data/dataset_3dmotion/motionx_finedance_aistpp_30fps_22joints/motionx_new_joint_vecs/humanml/005836",
    "/data/luomingshuang/data/dataset_3dmotion/motionx_finedance_aistpp_30fps_22joints/motionx_new_joint_vecs/humanml/M005836",
]

class MotionDatasetCB_LM(data.Dataset):
    def __init__(
        self,
        data_root,
        split,
        unit_length,
        mean,
        std,
        max_motion_length,
        min_motion_length,
        motionx_motion_code_path,
        motionx_text_path,
        finedance_motion_code_path,
        finedance_music_code_path,
        aistpp_motion_code_path,
        aistpp_motion_code_path,
        fps=30,
        stage='lm_pretrain',
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):       
        # Data mean and std
        self.mean = np.load(mean)
        self.std = np.load(std)

        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.window_size = window_size

        self.max_length = 30
        
        # Motion-x Data path
        motionx_split_file = pjoin(data_root, 'splits/motionx/{}.txt'.format(split))
        motionx_motion_cb_dir = motionx_motion_code_path
        motionx_text_dir = motionx_text_path

        # Finedance Data path
        finedance_split_file = pjoin(data_root, 'splits/finedance/{}.txt'.format(split))
        finedance_motion_cb_dir = finedance_motion_code_path
        finedance_music_cb_dir = finedance_music_code_path

        # Aistpp Data path
        aistpp_split_file = pjoin(data_root, 'splits/aistpp/{}.txt'.format(split))
        aistpp_motion_cb_dir = aistpp_motion_code_path
        aistpp_music_cb_dir = aistpp_music_code_path

        # Data id list
        new_name_list = [] 
        data_dict = {}

        with cs.open(motionx_split_file, "r") as f:
            for line in f.readlines():
                motionx_motion_cb_file = pjoin(motionx_motion_cb_dir, f'{line.strip()}.npy')
                if os.path.exists(motionx_motion_cb_file):
                    motionx_text_file = pjoin(motionx_text_dir, f'{line.strip()}.txt')
                    m_token_list = np.load(motionx_motion_cb_file)
                    with cs.open(motionx_text_file) as f:
                        text_data = []
                        flag = False
                        lines = f.readlines()
                        for line in lines:
                            try:
                                text_dict = {}
                                line_split = line.strip().split('#')
                                caption = line_split[0]
                                t_tokens = line_split[1].split(' ')
                                f_tag = float(line_split[2])
                                to_tag = float(line_split[3])
                                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                                text_dict['caption'] = caption
                                text_dict['tokens'] = t_tokens
                                if f_tag == 0.0 and to_tag == 0.0:
                                    flag = True
                                    text_data.append(text_dict)
                                else:
                                    m_token_list_new = [
                                        tokens[int(f_tag * fps / unit_length):int(to_tag * fps / unit_length)]
                                        for tokens in m_token_list
                                        if int(f_tag * fps / unit_length) <
                                        int(to_tag * fps / unit_length)
                                    ]

                                    if len(m_token_list_new) == 0:
                                        continue
                                    new_name = '%s_%f_%f' % (name, f_tag, to_tag)

                                    data_dict[new_name] = {'task': 't2m', 'm_token_list': m_token_list_new,
                                                           'text': [text_dict]}
                                    new_name_list.append(new_name)
                            except:
                                pass
                    if flag:
                        data_dict[name] = {
                            'task': 't2m',
                            'm_token_list': m_token_list,
                            'text': text_data
                        }
                        new_name_list.append(name)
                else:
                    continue

        with cs.open(finedance_split_file, "r") as f:
            for line in f.readlines():
                finedance_motion_cb_file = pjoin(finedance_motion_cb_dir, f'{line.strip()}.npy')
                if os.path.exists(finedance_motion_cb_file):
                    finedance_music_cb_file = pjoin(finedance_music_cb_dir, f'{line.strip()}.npy')
                    dance_token_list = np.load(finedance_motion_cb_file)
                    music_token_list = np.load(finedance_music_cb_file)
                    full_seq_len = dataset_opt.finedance_seq_len
                    each_seq_time = int(full_seq_len/dataset_opt.fps)
                    each_seq_tokens_num = int(each_seq_time*dataset_opt.num_tokens_each_seconds)
                    total_seq_time = int(dance_token_list.shape[0]/dataset_opt.num_tokens_each_seconds)
                    slide = dataset_opt.finedance_seq_len // dataset_opt.windows
                    nums = (dance_token_list.shape[0] - each_seq_tokens_num) // slide + 1 
                else:
                    continue

        with cs.open(aistpp_split_file, "r") as f:
            for line in f.readlines():
                self.id_list.append(pjoin(aistpp_motion_dir, line.strip()))

        self.id_list = [id_name for id_name in self.id_list if id_name not in ignore_list]
        print(len(self.id_list))
        # Debug mode
        if tiny or debug:
            enumerator = enumerate(tqdm(self.id_list))
            maxdata = 100
            subset = '_tiny'
        else:
            enumerator = enumerate(tqdm(self.id_list))
            maxdata = 1e10
            subset = ''

        new_name_list = []
        length_list = []
        motion_dict = {}
        
        for idx, name in enumerator:
            if len(new_name_list) > maxdata:
                break
            try:
                motion = np.load(name + ".npy")
                if motion.shape[0] > self.window_size:
                    motion_dict[name] = {
                        'motion': motion, 
                        "length": motion.shape[0]}
                    new_name_list.append(name)
                    length_list.append(motion.shape[0])
                    # print(len(length_list))
            except:
                pass
        print('the total number of motion samples for {} :'.format(split), len(new_name_list))

        self.motion_dict = motion_dict
        self.name_list = new_name_list
        self.nfeats = motion_dict[new_name_list[0]]['motion'].shape[1]

    def __len__(self):
        return len(self.name_list)

    def feats2joints(self, features):
        mean = torch.tensor(self.mean).to(features)
        std = torch.tensor(self.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, 22)

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.motion_dict[self.name_list[item]]
        motion, m_length = data["motion"], data["length"]
        
        # Z Normalization
        motion = (motion - self.mean) / self.std

        return name, motion, m_length, None, None, None, None,

def prepare_motion_data():
    motion_dataset_train = MotionDatasetCB_LM(
        data_root=dataset_opt.data_root,
        split='train',
        window_size=dataset_opt.window_size,
        mean=dataset_opt.mean,
        std=dataset_opt.std,
        max_motion_length=dataset_opt.max_motion_length,
        min_motion_length=dataset_opt.min_motion_length,
        fps=dataset_opt.fps,
        tmpFile=True,
        tiny=False,
        debug=False,)

    motion_dataset_val = MotionDatasetCB_LM(
        data_root=dataset_opt.data_root,
        split='val',
        window_size=dataset_opt.window_size,
        mean=dataset_opt.mean,
        std=dataset_opt.std,
        max_motion_length=dataset_opt.max_motion_length,
        min_motion_length=dataset_opt.min_motion_length,
        fps=dataset_opt.fps,
        tmpFile=True,
        tiny=False,
        debug=False,)

    motion_dataset_test = MotionDatasetCB_LM(
        data_root=dataset_opt.data_root,
        split='test',
        window_size=dataset_opt.window_size,
        mean=dataset_opt.mean,
        std=dataset_opt.std,
        max_motion_length=dataset_opt.max_motion_length,
        min_motion_length=dataset_opt.min_motion_length,
        fps=dataset_opt.fps,
        tmpFile=True,
        tiny=False,
        debug=False,)

    return motion_dataset_train, motion_dataset_val, motion_dataset_test

_, _, _ = prepare_motion_data()