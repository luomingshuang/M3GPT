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

class MotionDataset(data.Dataset):
    def __init__(
        self,
        data_root,
        split,
        window_size,
        mean,
        std,
        max_motion_length,
        min_motion_length,
        fps=30,
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
        motionx_motion_dir = pjoin(data_root, 'motionx_new_joint_vecs')

        # Finedance Data path
        finedance_split_file = pjoin(data_root, 'splits/finedance/{}.txt'.format(split))
        finedance_motion_dir = pjoin(data_root, 'finedance_new_joint_vecs')

        # Aistpp Data path
        aistpp_split_file = pjoin(data_root, 'splits/aistpp/{}.txt'.format(split))
        aistpp_motion_dir = pjoin(data_root, 'aistpp_new_joint_vecs')

        # Data id list
        self.id_list = [] 

        with cs.open(motionx_split_file, "r") as f:
            for line in f.readlines():
                self.id_list.append(pjoin(motionx_motion_dir, line.strip()))

        with cs.open(finedance_split_file, "r") as f:
            for line in f.readlines():
                self.id_list.append(pjoin(finedance_motion_dir, line.strip()))

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
                ## filter less than window_size
                if motion.shape[0] > self.window_size:
                    motion_dict[name] = {
                        'motion': motion, 
                        "length": motion.shape[0]}
                    new_name_list.append(name)
                    length_list.append(motion.shape[0])
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
    motion_dataset_train = MotionDataset(
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

    motion_dataset_val = MotionDataset(
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

    motion_dataset_test = MotionDataset(
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


def prepare_motion_vq_data_test_motion():

    motion_dataset_val = MotionDataset(
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

    motion_dataset_test = MotionDataset(
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

    return motion_dataset_val, motion_dataset_test