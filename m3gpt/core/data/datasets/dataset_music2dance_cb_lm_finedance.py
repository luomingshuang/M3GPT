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

class Music2DanceDatasetCB_Finedance(data.Dataset):
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
        finedance_music_code_path,
        aistpp_dance_code_path,
        aistpp_music_code_path,
        instructions,
        fps=30,
        sampling_rate=22050,
        only_finedance=False,
        tmpFile=True,
        tiny=False,
        debug=False,
        std_text=False,
        **kwargs,
    ):       
        # task name
        self.task_name = ginfo.task_name

        # Data mean and std
        self.mean = np.load(mean)
        self.std = np.load(std)

        self.fps = fps
        self.sampling_rate = sampling_rate
        self.motion_tokens_per_second = self.fps*1.0 / unit_length
        self.audio_tokens_per_second = self.sampling_rate*1.0 / 128

        self.stride = self.audio_tokens_per_second * 1.0 / self.motion_tokens_per_second
        self.win_motion_size = int(window_seconds * self.motion_tokens_per_second) # 15
        self.win_audio_size = int(window_seconds * self.audio_tokens_per_second) # 344
        
        data_dict = {}
        name_list = []

        # finedance Data path
        finedance_split_file = pjoin(data_root, 'splits/finedance/{}.txt'.format(split))
        finedance_dance_cb_dir = finedance_dance_code_path
        finedance_audio_cb_dir = finedance_music_code_path

        finedance_data_dict, finedance_name_list = self.load_audio_dance_finedance(finedance_split_file, finedance_dance_cb_dir, finedance_audio_cb_dir)
        print('finedance {} has {} motion files.'.format(split, len(finedance_name_list)))

        data_dict.update(finedance_data_dict)
        name_list = finedance_name_list

        self.data_dict = data_dict
        self.name_list = name_list

        task = ''
        self.instructions = json.load(open(instructions, 'r'))
        for task in self.instructions.keys():
            for subtask in self.instructions[task].keys():
                task = self.instructions[task][subtask]

        self.data_samples = []
    
        for name in tqdm(self.data_dict.keys()):
            m_token_list = self.data_dict[name]['m_token']
            a_token_list = self.data_dict[name]['a_token']
            for i in range(len(m_token_list) - self.win_motion_size):
                motion_token_sample = m_token_list[i:i+self.win_motion_size]
                audio_token_sample = a_token_list[int(i*self.stride):int(i*self.stride)+self.win_audio_size] 
                self.data_samples.append((motion_token_sample, audio_token_sample, task))
            
            self.data_samples.append((m_token_list[-self.win_motion_size:], a_token_list[-self.win_audio_size:], task))          

        print('the total number of data samples for {}: '.format(split), len(self.data_samples))

    def load_audio_dance_finedance(self, split_file, dance_dir, audio_dir):
        # Data id list
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        
        # data
        new_name_list = []
        data_dict = {}

        for _, name in enumerate(track(id_list)):
            try:
                # print(dance_dir, name)
                # import pdb; pdb.set_trace()
                m_token_list = np.load(pjoin(dance_dir, f'{name}.npy')) 
                m_token = m_token_list #[T//8]
                
                audio_name = name + '.npy'
                a_token = np.load(pjoin(audio_dir, audio_name)) #[T//256]

                m_token, a_token = self.align(m_token, a_token)

                data_dict[name] = {
                    'm_token': m_token,
                    'a_token': a_token,
                }
                new_name_list.append(name)

            except:
                print(pjoin(dance_dir, f'{name}.npy'))
                pass

        return data_dict, new_name_list

    def load_audio_dance_aistpp(self, split_file, dance_dir, audio_dir):
        # Data id list
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        
        # data
        new_name_list = []
        data_dict = {}

        for _, name in enumerate(track(id_list)):
            try:
                # print(dance_dir, name)
                # import pdb; pdb.set_trace()
                m_token_list = np.load(pjoin(dance_dir, f'{name}.npy')) 
                m_token = m_token_list #[T//8]
                
                audio_name = name.split('_')[4] + '.npy'
                a_token = np.load(pjoin(audio_dir, audio_name)) #[T//256]

                m_token, a_token = self.align(m_token, a_token)

                data_dict[name] = {
                    'm_token': m_token,
                    'a_token': a_token,
                }
                new_name_list.append(name)

            except:
                print(pjoin(dance_dir, f'{name}.npy'))
                pass

        return data_dict, new_name_list

    def align(self, motion, audio):
        # motion: [T1]; audio [T2]
        
        #print('---------- Align the frames of motion and audio ----------')
        t1 = motion.shape[0] // self.motion_tokens_per_second
        t2 = audio.shape[0] // self.audio_tokens_per_second

        min_seq_len = min(t1, t2)
        # print(f'motion -> {motion.shape}, ' +
        #       f'audio -> {audio.shape}, ' +
        #       f'min_seq_len -> {min_seq_len}')
        
        new_len1 = int(min_seq_len * self.motion_tokens_per_second)
        new_len2 = int(min_seq_len * self.audio_tokens_per_second)

        new_motion = motion[:new_len1]
        new_audio = audio[:new_len2]
        # print(f'new_motion -> {new_motion.shape}, ' +
        #       f'new_audio -> {new_audio.shape}')
        return new_motion, new_audio

    def feats2joints(self, features):
        mean = torch.tensor(self.mean).to(features)
        std = torch.tensor(self.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, 22)

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, item):
        m_tokens, a_tokens, task = self.data_samples[item][0], self.data_samples[item][1], self.data_samples[item][2]

        m_tokens_len = m_tokens.shape[0]
        a_tokens_len = a_tokens.shape[0]

        return None, a_tokens, m_tokens, m_tokens_len, a_tokens_len, task
        
        
        
        # return caption, m_tokens, m_tokens_len, None, None, None, None, all_captions, tasks


# def prepare_motion_data():
#     motion_dataset_train = MotionDatasetCB_LM(
#         data_root=dataset_opt.data_root,
#         split='train',
#         window_size=dataset_opt.window_size,
#         mean=dataset_opt.mean,
#         std=dataset_opt.std,
#         max_motion_length=dataset_opt.max_motion_length,
#         min_motion_length=dataset_opt.min_motion_length,
#         fps=dataset_opt.fps,
#         tmpFile=True,
#         tiny=False,
#         debug=False,)

#     motion_dataset_val = MotionDatasetCB_LM(
#         data_root=dataset_opt.data_root,
#         split='val',
#         window_size=dataset_opt.window_size,
#         mean=dataset_opt.mean,
#         std=dataset_opt.std,
#         max_motion_length=dataset_opt.max_motion_length,
#         min_motion_length=dataset_opt.min_motion_length,
#         fps=dataset_opt.fps,
#         tmpFile=True,
#         tiny=False,
#         debug=False,)

#     motion_dataset_test = MotionDatasetCB_LM(
#         data_root=dataset_opt.data_root,
#         split='test',
#         window_size=dataset_opt.window_size,
#         mean=dataset_opt.mean,
#         std=dataset_opt.std,
#         max_motion_length=dataset_opt.max_motion_length,
#         min_motion_length=dataset_opt.min_motion_length,
#         fps=dataset_opt.fps,
#         tmpFile=True,
#         tiny=False,
#         debug=False,)

#     return motion_dataset_train, motion_dataset_val, motion_dataset_test

# _, _, _ = prepare_motion_data()