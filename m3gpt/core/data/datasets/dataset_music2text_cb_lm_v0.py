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

class Music2TextDatasetCB_v0(data.Dataset):
    def __init__(
        self,
        ginfo,
        data_root,
        split,
        instructions,
        **kwargs,
    ):       
        # task name
        self.task_name = ginfo.task_name
        
        # MusicBench Data path
        musicbench_split_file = pjoin(data_root, 'splits/{}.txt'.format(split))

        # instruction 
        self.instructions = instructions

        # Data id list
        new_name_list = [] 
        data_dict = {}
        
        with cs.open(musicbench_split_file, "r") as f:
            for idx, line in enumerate(tqdm(f.readlines())):
                # print(line.split('  '))
                if len(line.split('  ')) == 2:

                    music_token_file, text = line.split('  ')
                    text = text.rstrip('\n')
                    ## 172.5 tokens / seconds
                    if os.path.exists(music_token_file):
                        name = music_token_file.split('/')[-1][:-4]
                        a_token = np.load(music_token_file)[0][:int(9.91*172)]
                        data_dict[name] = {
                            'task': 'a2t',
                            'a_token': a_token,
                            'text': text
                        }
                        new_name_list.append(name)
                    else:
                        continue
                else:
                    continue
                # print(data_dict)

        new_name_list = [id_name for id_name in new_name_list]
        print('the total number of motion samples for {} :'.format(split), len(new_name_list))

        self.data_dict = data_dict
        self.name_list = new_name_list
        self.instructions = json.load(open(self.instructions, 'r'))
        self.tasks = []
        for task in self.instructions.keys():
            for subtask in self.instructions[task].keys():
                self.tasks.append(self.instructions[task][subtask])

    def __len__(self):
        return len(self.name_list)*len(self.tasks)

    def __getitem__(self, item):
        data_idx = item % len(self.name_list)
        task_idx = item // len(self.name_list)

        data = self.data_dict[self.name_list[data_idx]]
        a_tokens, text = data['a_token'], data['text']

        a_tokens_len = a_tokens.shape[0]

        tasks = self.tasks[task_idx]

        return text, a_tokens, None, None, a_tokens_len, tasks


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