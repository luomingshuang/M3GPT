import os
import rich
import random
import codecs as cs
import numpy as np

import torch
from torch.utils import data
from tqdm import tqdm
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

class MotionDatasetVQ(data.Dataset):
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
        super().__init__()
        # data mean and std
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

        ## increase finedance samples for finedance
        if split == 'train':
            increasing_times = 10
        else:
            increasing_times = 1
            
        with cs.open(finedance_split_file, "r") as f:
            for line in f.readlines():
                for i in range(increasing_times):
                    self.id_list.append(pjoin(finedance_motion_dir, line.strip()))

        with cs.open(aistpp_split_file, "r") as f:
            for line in f.readlines():
                for i in range(increasing_times):
                    self.id_list.append(pjoin(aistpp_motion_dir, line.strip()))

        self.id_list = [id_name for id_name in self.id_list if id_name not in ignore_list]

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
        self.lengths = []
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
                   
            except:
                pass
        print('the total number of motion samples for {} :'.format(split), len(new_name_list))
        
        augment = True
        if augment:
            for idx, name in enumerate(self.id_list):
                if len(new_name_list) > maxdata:
                    break
                try:
                    motion = np.load(name + ".npy")
                    # clip long to short, increasing dance samples
                    if motion.shape[0] >= 640:
                        num_clips = motion.shape[0] // 320
                        # print(num_clips)
                        for i in range(num_clips):
                            name_clip = name + '_{}'.format(i)
                            if i == 0:
                                motion_clip = motion[:300*(i+1)]
                            elif 0 < i <= num_clips - 2:
                                motion_clip = motion[300*i:300*(i+1)]
                            else:
                                motion_clip = motion[300*i:]
                            # print(name_clip)
                            if motion_clip.shape[0] > self.window_size:
                                motion_dict[name_clip] = {
                                    'motion': motion_clip, 
                                    "length": motion_clip.shape[0]}
                                new_name_list.append(name_clip)
                                length_list.append(motion_clip.shape[0])
                except:
                    pass

        print('the total number of motion samples for {} :'.format(split), len(new_name_list))
        self.motion_dict = motion_dict
        self.name_list = new_name_list
        self.length_arr = np.array(length_list)
        self.nfeats = motion_dict[new_name_list[0]]['motion'].shape[1]
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def feats2joints(self, features):
        mean = torch.tensor(self.mean).to(features)
        std = torch.tensor(self.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, 22)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.motion_dict[self.name_list[item]]
        motion, m_length = data["motion"], data["length"]

        idx = random.randint(0, motion.shape[0] - self.window_size)
        motion = motion[idx:idx + self.window_size]

        motion = (motion - self.mean) / self.std

        return None, motion, m_length, None, None, None, None,

def prepare_motion_vq_data():
    motion_vq_dataset_train = MotionDatasetVQ(
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

    motion_vq_dataset_val = MotionDatasetVQ(
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

    motion_vq_dataset_test = MotionDatasetVQ(
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

    return motion_vq_dataset_train, motion_vq_dataset_val, motion_vq_dataset_test

def prepare_motion_vq_data_test_motion():
    motion_vq_dataset_val = MotionDatasetVQ(
        data_root=dataset_opt.data_root,
        split='val',
        window_size=dataset_opt.window_size*2,
        mean=dataset_opt.mean,
        std=dataset_opt.std,
        max_motion_length=dataset_opt.max_motion_length,
        min_motion_length=dataset_opt.min_motion_length,
        fps=dataset_opt.fps,
        tmpFile=True,
        tiny=False,
        debug=False,)

    motion_vq_dataset_test = MotionDatasetVQ(
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

    return motion_vq_dataset_val, motion_vq_dataset_test