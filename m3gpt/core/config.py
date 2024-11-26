import yaml
import logging
import numpy as np
from easydict import EasyDict as edict
import copy
import re

from core import distributed_utils as dist

from .utils import printlog


task_specific_param = ['lm', 'dataset', 'sampler', 'lr_scheduler', 'optimizer',
                       'extra', 'evaluation', 'model_entry_type', 'load_ignore', 'ckpt_task_id']

loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

def flat(nums):
    res = []
    for i in nums:
        if isinstance(i, list):
            res.extend(flat(i))
        else:
            res.append(i)
    return res

def specific_group_split(args, group_spec, share_lm_group_ids):
    ## sanity check
    assert type(group_spec) is list
    assert all(map(lambda x: type(x) is int, group_spec))

    num_groups = len(group_spec)
    splits = np.sum(group_spec)

    world_size = ''
    rank = ''
    if args.method == "slurm":
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = dist.get_world_size_no_slurm()
        rank = dist.get_local_rank_no_slurm()

    # import pdb; pdb.set_trace()

    assert world_size % splits == 0, f"{world_size} % {splits}"
    unit = int(world_size / splits)

    ## split
    group_sizes = [x*unit for x in group_spec]  # [8,8,8] / [32, 16]
    groups = []
    roots = []
    last = 0
    task_info = edict()
    all_ranks = []
    for i,gs in enumerate(group_sizes):
        ranks = list(map(int, np.arange(last, last+gs)))  #[0...8], [9...15], ...
        groups.append(dist.new_group(ranks=ranks))
        roots.append(last) # 0, 8, 16
        all_ranks.append(ranks)
        if rank in ranks:  # if current gpu rank in traversed rank task group
            printlog(f">> task_info.group[{i}] ranks {ranks}")
            task_info.group = groups[-1]  # subordinate to what group
            task_info.task_size = gs  # 8
            task_info.task_id = i
            task_info.task_rank = rank - last
            task_info.task_root_rank = last
        last += gs
    task_info.root_group = dist.new_group(ranks=roots)
    printlog(f">> task_info.root_group ranks {roots}")
    task_info.task_sizes = group_sizes
    task_info.task_root_ranks = roots
    task_info.task_num = num_groups

    ## lm_backbone_group spec
    if share_lm_group_ids is not None:  # *[0,0,0]*(default) | [0,1,0]task ids
        # group size must equal within a share_group
        lmshareid2idx = {}
        for idx, this_id in enumerate(share_lm_group_ids):
            if this_id not in lmshareid2idx:
                lmshareid2idx[this_id] = list()
            lmshareid2idx[this_id].append(idx)  # {0: [0,1,2]}| {0: [0,2], 1: [1]}

        ## create lm share group
        for idxs in lmshareid2idx.values():  # idxs = [0, 1, 2]
            this_group_ranks = flat([all_ranks[i] for i in idxs])
            this_share_group = dist.new_group(ranks=this_group_ranks)
            this_group_size = len(this_group_ranks)
            if rank in this_group_ranks:
                task_info.lm_share_group = this_share_group
                printlog(f">> task_info.lm_share_group[{idxs}] ranks {this_group_ranks}")
                task_info.lm_group_size = len(lmshareid2idx)
                task_info.lm_task_size = len(lmshareid2idx) * this_group_size
                task_info.lm_task_rank = np.sum(rank < np.array(this_group_ranks))

    return task_info

class Config(object):

    def __init__(self, config_file, args, noginfo=False, spec_ginfo_index=None):
        with open(config_file) as f:
            config = yaml.load(f, Loader=loader)
        self.config_path = config_file

        world_size = ''
        rank = ''
        if args.method == "slurm":
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = dist.get_world_size_no_slurm()
            rank = dist.get_local_rank_no_slurm()

        if noginfo:
            ginfo = None
        else:  # cherrypick from tasks
            tasks = config['tasks']

            num_tasks = len(tasks)
            if spec_ginfo_index is not None:
                assert spec_ginfo_index < len(tasks), \
                'spec_ginfo_index={} is larger than num_tasks={}'.format(spec_ginfo_index, len(tasks))
                tmp_config = copy.deepcopy(config)
                config['tasks'] = dict()
                config['tasks'][0] = tmp_config['tasks'][spec_ginfo_index]
                config['tasks'][0]['gres_ratio'] = 1
                tasks = config['tasks']
                num_tasks = len(tasks)

            # parse task_common and assign to each task
            task_common = config.get('task_common', None)
            if task_common is not None:
                for i in range(num_tasks):
                    for k,v in task_common.items():
                        if not k in tasks[i]:
                            printlog('setting {} to {} for task {}'.format(k, v, i))
                            tasks[i][k] = v

            group_spec = [tasks[i].get('gres_ratio',1) for i in range(num_tasks)]
            ## share group spec
            if config['common'].get('share_lm_group', False):
                share_lm_group_ids = config['common']['share_lm_group'][:num_tasks]
            else:
                share_lm_group_ids = [0 for i in range(num_tasks)]  # hardcoded prior

            ginfo = specific_group_split(args, group_spec, share_lm_group_ids)
            loss_weight_sum = float(np.sum(np.array([task['loss_weight'] for task in tasks.values()])))
            ginfo.task_name = tasks[ginfo.task_id]['name']
            ginfo.task_names = [tasks[i]['name'] for i in range(ginfo.task_num)]
            ginfo.task_weight = float(tasks[ginfo.task_id]['loss_weight']) / loss_weight_sum
            ginfo.task_type = tasks[ginfo.task_id].get('type', 'normal')
            ginfo.task_types = [tasks[i].get('type', 'normal') for i in range(ginfo.task_num)]
            ginfo.task_random_seed = tasks[ginfo.task_id].get('random_seed', 0)

            for p in task_specific_param:
                if p in config['tasks'][ginfo.task_id]:
                    config['common'][p] = config['tasks'][ginfo.task_id][p]
                    printlog('{} of task{} has been overided to {}'.format(p, ginfo.task_id, config['common'][p]))

        logger = logging.getLogger('global_logger')

        self.world_size = world_size
        self.rank = rank
        self.ginfo = ginfo
        self.config = config
        self.config_file = config_file
