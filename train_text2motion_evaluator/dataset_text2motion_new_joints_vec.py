import os
import rich
import random
import pickle
import codecs as cs
import numpy as np
from tqdm import tqdm
import spacy
import json

import torch
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin

from scripts.motion_process import process_file, recover_from_ric
# from utils_motion.word_vectorizer import WordVectorizer

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

class Text2MotionDatasetCB(data.Dataset):
    def __init__(
        self,
        task_name,
        data_root,
        split,
        unit_length,
        mean,
        std,
        w_vectorizer,
        motionx_motion_code_path,
        motionx_text_path,
        instructions,
        fps=30,
        tmpFile=True,
        tiny=False,
        debug=False,
        std_text=False,
        **kwargs,
    ):       
        # task name
        self.task_name = task_name

        # Data mean and std
        self.mean = np.load(mean)
        self.std = np.load(std)
        # Using the same mean and std
        # self.mean_eval = np.load('/data/luomingshuang/data/dataset_3dmotion/motionx_finedance_aistpp_30fps_22joints/Mean_test.npy')
        # self.std_eval = np.load('/data/luomingshuang/data/dataset_3dmotion/motionx_finedance_aistpp_30fps_22joints/Std_test.npy')

        self.max_length = 30
        self.unit_length = unit_length
        
        # self.w_vectorizer = WordVectorizer(w_vectorizer, "our_vab")

        # Motion-x Data path
        motionx_split_file = pjoin(data_root, 'splits/motionx/{}.txt'.format(split))
        motionx_motion_feats_dir = pjoin(data_root, 'motionx_new_joint_vecs')
        # motionx_motion_feats_dir = '/home/luomingshuang/data/dataset_3dmotion/motion-x/motion_data/body_feats/new_joints'
        motionx_motion_cb_dir = motionx_motion_code_path
        motionx_text_dir = motionx_text_path

        # instruction 
        self.instructions = instructions

        # Data id list
        new_name_list = [] 
        data_dict = {}
        
        max_motion_length = 0
        num_motion_length_over_64 = 0
        num_motion_length_over_128 = 0
        num_motion_length_over_192 = 0
        num_motion_length_over_256 = 0
        num_motion_length_over_15 = 0
        with cs.open(motionx_split_file, "r") as f:
            lines = f.readlines()# [:2000]
            print('the ori total number of motion samples for {} :'.format(split), len(lines))
            for idx, line in enumerate(tqdm(lines)):
                name = line.strip()
                motionx_motion_cb_file = pjoin(motionx_motion_cb_dir, f'{line.strip()}.npy')
                motionx_motion_feat_file = pjoin(motionx_motion_feats_dir, f'{line.strip()}.npy')
                if os.path.exists(motionx_motion_cb_file): # and np.load(motionx_motion_cb_file).shape[0] <= 128:
                    # print(np.load(motionx_motion_cb_file).shape[0])
                    motionx_text_file = pjoin(motionx_text_dir, f'{line.strip()}.txt')
                    m_token = np.load(motionx_motion_cb_file)[:128]
                    m_feats = np.load(motionx_motion_feat_file)[:4*128]
                    m_token_list = [m_token]
                    
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
                    if m_token.shape[0] >= 15:
                        num_motion_length_over_15 += 1

                    with cs.open(motionx_text_file) as f:
                        text_data = []
                        flag = False
                        lines = f.readlines()
                        for line in lines:
                            if '#' in line:
                                try:
                                    text_dict = {}
                                    line_split = line.strip().split('#')
                                    caption = line_split[0]
                                    t_tokens = line_split[1].split(' ')
                                    f_tag = float(line_split[2])
                                    to_tag = float(line_split[3])
                                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                                    to_tag = 0.0 if np.isnan(to_tag) else to_tag

                                    text_tokens = caption.split(' ')
                                    text_tokens = [text_token for text_token in text_tokens if text_token != '']
                                    caption = ' '.join(text_tokens[:50])
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
                                        'length': len(m_feats), 'm_feats': m_feats, 'text': [text_dict]}
                                        new_name_list.append(new_name)
                                except:
                                    pass
                            else:
                                try:
                                    # if '#' in line:
                                    #     print(line)
                                    text_dict = {}
                                    line_split = line.strip()
                                    caption = line_split
                                    t_tokens = line_split[:-1].split(' ')
                                    f_tag = 0.0
                                    to_tag = 0.0

                                    text_tokens = caption.split(' ')
                                    text_tokens = [text_token for text_token in text_tokens if text_token != '']
                                    caption = ' '.join(text_tokens[:50])
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
                                        'length': len(m_feats), 'm_feats': m_feats, 'text': [text_dict]}
                                        new_name_list.append(new_name)
                                except:
                                    pass

                    if flag:
                        # print(m_token_list)
                        data_dict[name] = {
                            'task': 't2m',
                            'm_token_list': m_token_list,
                            'length': len(m_feats),
                            'm_feats': m_feats,
                            'text': text_data
                        }
                        new_name_list.append(name)
                else:
                    continue

        new_name_list = [id_name for id_name in new_name_list if id_name not in ignore_list]
        print('the total number of motion samples for {} :'.format(split), len(new_name_list))
        print('the max motion tokens length for {} :'.format(split), max_motion_length)
        print('the number of motion tokens whose length over 15 for {} :'.format(split), num_motion_length_over_15)
        print('the number of motion tokens whose length over 64 for {} :'.format(split), num_motion_length_over_64)
        print('the number of motion tokens whose length over 128 for {} :'.format(split), num_motion_length_over_128)
        print('the number of motion tokens whose length over 192 for {} :'.format(split), num_motion_length_over_192)
        print('the number of motion tokens whose length over 256 for {} :'.format(split), num_motion_length_over_256)

        self.data_dict = data_dict
        self.name_list = new_name_list
        self.nlp = spacy.load('en_core_web_sm')
        self.std_text = std_text
        self.instructions = os.path.join(self.instructions, '{}_template_pretrain.json'.format(self.task_name))
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

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.mean).to(features)
        ori_std = torch.tensor(self.std).to(features)
        eval_mean = torch.tensor(self.mean_eval).to(features)
        eval_std = torch.tensor(self.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def __getitem__(self, item):
        data_idx = item % len(self.name_list)
        task_idx = item // len(self.name_list)

        data = self.data_dict[self.name_list[data_idx]]
        m_token_list, m_feats, text_list = data['m_token_list'], data['m_feats'], data['text']

        m_feats = (m_feats - self.mean) / self.std

        m_tokens = random.choice(m_token_list)
        text_data = random.choice(text_list)
        caption = text_data['caption']

        m_length = data['length']
        # print(m_length)
        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'
        
        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(m_feats) - m_length)
        m_feats = m_feats[idx:idx+m_length]
        # print(m_feats.shape)

        mask = torch.ones(len(m_feats)).int()

        return caption, m_feats, mask


import options_t2m as opt

def prepare_text2motion_data(task_name, split):
    text2motion_data = Text2MotionDatasetCB(
        task_name=task_name,
        data_root=opt.data_root,
        split=split,
        unit_length=opt.unit_length,
        mean=opt.mean,
        std=opt.std,
        w_vectorizer=opt.w_vectorizer,
        motionx_motion_code_path=opt.motionx_motion_code_path,
        motionx_text_path=opt.motionx_text_path,
        instructions=opt.instructions,
        fps=opt.fps,
        tmpFile=True,
        tiny=False,
        debug=False,)

    return text2motion_data

# _ = prepare_text2motion_data('t2m', 'val')