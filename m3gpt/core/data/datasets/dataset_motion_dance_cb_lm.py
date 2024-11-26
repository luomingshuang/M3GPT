import os
import rich
import random
import pickle
import codecs as cs
import numpy as np
from tqdm import tqdm
import json

import torch
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin

from .scripts.motion_process import process_file, recover_from_ric

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

class MotionDanceDatasetCB(data.Dataset):
    def __init__(
        self,
        ginfo,
        data_root,
        split,
        unit_length,
        window_seconds,
        mean,
        std,
        finedance_dance_code_path,
        aistpp_dance_code_path,
        motionx_motion_code_path,
        instructions,
        fps=30,
        sampling_rate=22050,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):       
        # task name
        self.task_name = ginfo.task_name

        # Data mean and std
        self.mean = np.load(mean)
        self.std = np.load(std)

        self.max_length = 30
        
        # Motion-x Data path
        motionx_split_file = pjoin(data_root, 'splits/motionx/{}.txt'.format(split))
        motionx_motion_cb_dir = motionx_motion_code_path

        # finedance Data path
        finedance_split_file = pjoin(data_root, 'splits/finedance/{}.txt'.format(split))
        finedance_dance_cb_dir = finedance_dance_code_path

        # aistpp Data path
        aistpp_split_file = pjoin(data_root, 'splits/aistpp/{}.txt'.format(split))
        aistpp_dance_cb_dir = aistpp_dance_code_path

        # instruction 
        self.instructions = instructions

        # Data id list
        new_name_list = [] 
        data_dict = {}
        
        total_motion_dance_sample = 0

        # add finedance train dance data
        self.window_seconds = window_seconds
        self.fps = fps
        self.sampling_rate = sampling_rate
        self.motion_tokens_per_second = self.fps*1.0 / unit_length
        self.audio_tokens_per_second = self.sampling_rate*1.0 / 128
        self.seg_dance_length = int(self.window_seconds*self.motion_tokens_per_second)

        with cs.open(finedance_split_file, "r") as f:
            for idx, line in enumerate(tqdm(f.readlines())):
                name = line.strip()
                finedance_motion_cb_file = pjoin(finedance_dance_cb_dir, f'{name}.npy')
                if os.path.exists(finedance_motion_cb_file):
                    m_token = np.load(finedance_motion_cb_file)
                    data_dict[name] = {'m_token_list': m_token, 'type': 'dance'}
                    new_name_list.append(name)
                    total_motion_dance_sample += 1
                else:
                    continue
        print(f'the number of dance sample in finedance for prediction and inbetween is {total_motion_dance_sample}')

        # add aistpp train dance data
        with cs.open(aistpp_split_file, "r") as f:
            for idx, line in enumerate(tqdm(f.readlines())):
                name = line.strip()
                aistpp_motion_cb_file = pjoin(aistpp_dance_cb_dir, f'{name}.npy')
                if os.path.exists(aistpp_motion_cb_file):
                    m_token = np.load(aistpp_motion_cb_file)
                    data_dict[name] = {'m_token_list': m_token, 'type': 'dance'}
                    new_name_list.append(name)
                    total_motion_dance_sample += 1
                else:
                    continue
        print(f'the number of dance sample in finedance and aistpp for prediction and inbetween is {total_motion_dance_sample}')

        # add motion-x train motion data
        max_motion_length = 0
        num_motion_length_over_64 = 0
        num_motion_length_over_128 = 0
        num_motion_length_over_192 = 0
        num_motion_length_over_256 = 0
        with cs.open(motionx_split_file, "r") as f:
            for idx, line in enumerate(tqdm(f.readlines())):
                name = line.strip()
                motionx_motion_cb_file = pjoin(motionx_motion_cb_dir, f'{name}.npy')
                if os.path.exists(motionx_motion_cb_file):
                    m_token = np.load(motionx_motion_cb_file)
                    
                    if m_token.shape[0] > max_motion_length:
                        max_motion_length = m_token.shape[0]
                    if m_token.shape[0] > 256:
                        num_motion_length_over_256 += 1
                    if m_token.shape[0] > 192:
                        num_motion_length_over_192 += 1
                    if m_token.shape[0] > 128:
                        num_motion_length_over_128 += 1
                    if m_token.shape[0] > 64:
                        num_motion_length_over_64 += 1

                    data_dict[name] = {'m_token_list': m_token, 'type':'motion'}
                    new_name_list.append(name)
                    total_motion_dance_sample += 1
                else:
                    continue
        print(f'the number of dance sample (finedance-aistpp-motionx) for prediction and inbetween is {total_motion_dance_sample}')

        new_name_list = [id_name for id_name in new_name_list if id_name not in ignore_list]
        print('the total number of motion samples for {} :'.format(split), len(new_name_list))
        print('the max motion tokens length for {} :'.format(split), max_motion_length)
        print('the number of motion tokens whose length over 64 for {} :'.format(split), num_motion_length_over_64)
        print('the number of motion tokens whose length over 128 for {} :'.format(split), num_motion_length_over_128)
        print('the number of motion tokens whose length over 192 for {} :'.format(split), num_motion_length_over_192)
        print('the number of motion tokens whose length over 256 for {} :'.format(split), num_motion_length_over_256)


        self.data_dict = data_dict
        self.name_list = new_name_list
        self.instructions = json.load(open(self.instructions, 'r'))
        self.tasks = []
        for task in self.instructions.keys():
            for subtask in self.instructions[task].keys():
                self.tasks.append(self.instructions[task][subtask])

    def __len__(self):
        return len(self.name_list)*len(self.tasks)

    def feats2joints(self, features):
        mean = torch.tensor(self.mean).to(features)
        std = torch.tensor(self.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, 22)

    def __getitem__(self, item):
        data_idx = item % len(self.name_list)
        task_idx = item // len(self.name_list)

        data = self.data_dict[self.name_list[data_idx]]
        m_token_list = data['m_token_list']
        if data['type'] == 'dance' and m_token_list.shape[0]>self.seg_dance_length:
            idx = random.randint(0, m_token_list.shape[0] - self.seg_dance_length)
            m_token_list = m_token_list[idx : idx + self.seg_dance_length]            

        m_tokens = m_token_list

        coin = np.random.choice([False, False, True])

        if coin:
            # drop one token at the head or tail
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]

        m_tokens_len = m_tokens.shape[0]

        tasks = self.tasks[task_idx]

        return None, None, m_tokens, m_tokens_len, None, tasks